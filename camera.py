import threading
import cv2
import time
import os
import json
from datetime import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from ultralytics import YOLO # Necesario para obtener .names

class CameraThread(threading.Thread):
    def __init__(self, camera_config, yolo_model, static_path, event_sync_queue):
        super().__init__()
        self.daemon = True # El hilo morirá si el programa principal termina
        self.config = camera_config
        self.yolo_model = yolo_model # Modelo YOLO inyectado
        self.static_path = static_path # Path estático inyectado
        self.event_sync_queue = event_sync_queue # Cola para sincronización
        
        # Mapeo de clases de COCO
        self.coco_classes = self.yolo_model.names
        self.id = camera_config['id']
        self.name = camera_config['name']
        
        self.is_running = True
        self.last_frame = None
        self.lock = threading.Lock()
        self.detected_classes_in_frame = set()
        self.motion_detected = False
        
        # Opciones de configuración
        self.show_bbox = self.config.get('show_bbox', True)
        self.motion_detection_enabled = self.config.get('motion_detection_enabled', False)
        self.show_motion_contours = self.config.get('show_motion_contours', False)
        self.motion_sensitivity = self.config.get('motion_sensitivity', 1000)
        self.prev_gray_frame = None

        # Lógica de sesión de eventos
        self.similarity_threshold = 0.90
        self.active_event_session = False
        self.session_start_time = 0
        self.session_timeout = 5 # (segundos)
        self.session_reference_frame_gray = None
        self.session_best_event_data = None

        print(f"Inicializando cámara: {self.name}")
        # Crear carpeta para la cámara
        self.output_dir = os.path.join(self.static_path, str(self.id))
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        source = self.config['source']
        if self.config['type'] == 'local':
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la fuente de video para '{self.name}' en '{source}'.")
            return

        while self.is_running:
            if not self.config.get('enabled', False):
                time.sleep(1)
                continue

            success, frame = cap.read()
            if not success:
                print(f"No se pudo leer el frame de '{self.name}'. Reintentando en 5 segundos...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            # --- Detección de Movimiento ---
            frame_with_motion = self.detect_motion(frame)

            # --- Detección YOLO ---
            detect_classes = self.config.get('detect_classes')
            results = None
            if detect_classes:
                results = self.yolo_model.predict(source=frame, classes=detect_classes, verbose=False, device='cpu')[0]
                if results.boxes.cls.numel() > 0:
                    self.detected_classes_in_frame = set(results.boxes.cls.cpu().numpy().astype(int))
                else:
                    self.detected_classes_in_frame = set()
            else:
                 self.detected_classes_in_frame = set()
            
            # --- Gestión de Sesiones de Eventos ---
            self.manage_event_session(frame, results)

            # --- Preparar Frame para Streaming ---
            if results and self.show_bbox:
                processed_frame = results.plot()
            else:
                processed_frame = frame_with_motion # Muestra contornos de movimiento si están activos

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                with self.lock:
                    self.last_frame = buffer.tobytes()

        cap.release()
        print(f"Hilo de la cámara {self.name} detenido.")

    def detect_motion(self, frame):
        self.motion_detected = False
        frame_with_contours = frame.copy()
        
        if not self.motion_detection_enabled:
            return frame_with_contours # Devuelve el frame original

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if self.prev_gray_frame is not None:
            frame_delta = cv2.absdiff(self.prev_gray_frame, gray_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > self.motion_sensitivity:
                    self.motion_detected = True
                    if self.show_motion_contours:
                        cv2.drawContours(frame_with_contours, [contour], -1, (0, 0, 255), 2)
                    else:
                        break # Un contorno es suficiente si no necesitamos dibujarlos todos
            
        self.prev_gray_frame = gray_frame
        return frame_with_contours

    def get_dominant_color(self, image_crop):
        if image_crop.size == 0:
            return None
        image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pixels = image_crop_rgb.reshape((-1, 3))
        
        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_color_rgb = kmeans.cluster_centers_[unique[np.argmax(counts)]]
        
        return '#%02x%02x%02x' % (int(dominant_color_rgb[0]), int(dominant_color_rgb[1]), int(dominant_color_rgb[2]))

    def manage_event_session(self, frame, results):
        current_time = time.time()
        is_valid_event_trigger = self.motion_detected and results and results.boxes.cls.numel() > 0

        # --- Finalizar una sesión activa ---
        if self.active_event_session:
            if not is_valid_event_trigger or (current_time - self.session_start_time > self.session_timeout):
                print(f"[{self.name}] Fin de sesión de evento. Guardando el mejor frame.")
                self.save_event(self.session_best_event_data)
                self.active_event_session = False
                self.session_best_event_data = None
                self.session_reference_frame_gray = None
                return

        # --- Iniciar o continuar una sesión ---
        if is_valid_event_trigger:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.active_event_session:
                # Iniciar NUEVA sesión
                print(f"[{self.name}] Nueva sesión de evento iniciada.")
                self.active_event_session = True
                self.session_reference_frame_gray = frame_gray
                self.session_best_event_data = (frame.copy(), results)
            else:
                # Continuar sesión existente
                similarity = ssim(self.session_reference_frame_gray, frame_gray)
                if similarity < self.similarity_threshold:
                    # Cambio de escena: Guardar evento anterior e iniciar nueva sesión
                    print(f"[{self.name}] Cambio de escena (Sim: {similarity:.2f}). Guardando sesión anterior.")
                    self.save_event(self.session_best_event_data)
                    
                    print(f"[{self.name}] Iniciando nueva sesión con el frame actual.")
                    self.session_reference_frame_gray = frame_gray
                    self.session_best_event_data = (frame.copy(), results)
                # else: Imagen similar, no hacer nada, solo extender la sesión

            self.session_start_time = current_time # Actualizar última actividad

    def save_event(self, event_data):
        if not event_data:
            return
        
        frame, results = event_data
        timestamp = datetime.now()
        base_filename = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{base_filename}.jpg"
        json_filename = f"{base_filename}.json"
        image_path = os.path.join(self.output_dir, image_filename)
        json_path = os.path.join(self.output_dir, json_filename)
        
        cv2.imwrite(image_path, frame)
        print(f"Evento guardado: {image_path}")

        # --- Enviar evento a la cola de sincronización ---
        _, buffer = cv2.imencode('.jpg', frame)
        sync_payload = {
            "camera_name": self.name,
            "camera_id": self.id,
            "frame_bytes": buffer.tobytes()
        }
        self.event_sync_queue.put(sync_payload)
        print(f"[{self.name}] Evento encolado para sincronización.")

        objects_list = []
        objects_count = {}
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = self.coco_classes[class_id]
            bbox = [int(coord) for coord in box.xyxy[0]]
            x1, y1, x2, y2 = bbox
            object_crop = frame[y1:y2, x1:x2]

            object_data = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf),
                "bbox": bbox
            }

            if class_name == 'person':
                mid_y = y1 + (y2 - y1) // 2
                upper_crop = frame[y1:mid_y, x1:x2]
                lower_crop = frame[mid_y:y2, x1:x2]
                object_data['appearance'] = {
                    "upper_body": {
                        "bbox": [x1, y1, x2, mid_y],
                        "dominant_color_hex": self.get_dominant_color(upper_crop)
                    },
                    "lower_body": {
                        "bbox": [x1, mid_y, x2, y2],
                        "dominant_color_hex": self.get_dominant_color(lower_crop)
                    }
                }
            else:
                object_data['dominant_color_hex'] = self.get_dominant_color(object_crop)

            objects_list.append(object_data)
            objects_count[class_name] = objects_count.get(class_name, 0) + 1

        event_json_data = {
            "timestamp_utc": timestamp.isoformat(),
            "camera_id": self.id,
            "camera_name": self.name,
            "image_path": os.path.join(str(self.id), image_filename), # Ruta web relativa
            "objects": objects_list,
            "objects_count": objects_count
        }

        with open(json_path, 'w') as f:
            json.dump(event_json_data, f, indent=2)

    def stop(self):
        self.is_running = False

    def get_frame(self):
        with self.lock:
            return self.last_frame

    def update_config(self, key, value):
        with self.lock:
            if key == 'show_bbox':
                self.show_bbox = value
            elif key == 'show_motion':
                self.show_motion_contours = value
            elif key == 'motion_sensitivity':
                self.motion_sensitivity = int(value)
                self.config[key] = int(value)
            else:
                self.config[key] = value