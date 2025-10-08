from flask import Flask,Response, render_template, request, jsonify
import cv2
import yaml
import time
import os
import json
import threading
from ultralytics import YOLO
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# --- Carga del modelo YOLO ---
model = YOLO("yolov8n.pt")

# --- Gestión de Configuración ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# --- Clase para gestionar cada cámara en un hilo separado ---
class CameraThread(threading.Thread):
    def __init__(self, camera_config):
        super().__init__()
        self.config = camera_config
        # Añadimos el mapeo de clases de COCO para tener los nombres
        self.coco_classes = model.names
        self.id = camera_config['id']
        self.name = camera_config['name']
        self.is_running = True
        self.last_frame = None
        self.lock = threading.Lock()
        self.detected_classes_in_frame = set() # Clases detectadas en el último frame
        self.motion_detected = False
        self.show_bbox = self.config.get('show_bbox', True) # Nueva opción
        self.motion_detection_enabled = self.config.get('motion_detection_enabled', False)
        self.show_motion_contours = self.config.get('show_motion_contours', False)
        self.motion_sensitivity = self.config.get('motion_sensitivity', 1000)
        self.prev_gray_frame = None
        self.last_event_time = 0
        self.event_cooldown = 5 # Cooldown de 5 segundos entre eventos para no saturar
        print(f"Inicializando cámara: {self.name}")
        # Crear carpeta para la cámara si no existe
        self.output_dir = os.path.join(STATIC_PATH, str(self.id))
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
            self.motion_detected = False
            frame_with_motion = frame.copy() # Copiamos el frame para dibujar sobre él si es necesario
            if self.motion_detection_enabled:
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
                                cv2.drawContours(frame_with_motion, [contour], -1, (0, 0, 255), 2)
                            break # Un contorno es suficiente para detectar movimiento
                self.prev_gray_frame = gray_frame

            # Realizar detección con YOLO si hay clases para detectar
            detect_classes = self.config.get('detect_classes')
            current_detections = set()
            if detect_classes:
                # Usamos el frame original para la predicción para no tener los contornos de movimiento
                results = model.predict(source=frame, classes=detect_classes, verbose=False)[0]
                # Guardamos las clases detectadas en este frame
                if results.boxes.cls.numel() > 0:
                    current_detections = set(results.boxes.cls.cpu().numpy().astype(int))

                # --- Lógica para guardar eventos ---
                self.save_event_if_needed(frame, results)

                # Dibujar bounding boxes solo si la opción está activada
                processed_frame = results.plot() if self.show_bbox else frame_with_motion
            else:
                processed_frame = frame_with_motion

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                with self.lock:
                    self.last_frame = buffer.tobytes()
                    self.detected_classes_in_frame = current_detections

        cap.release()
        print(f"Hilo de la cámara {self.name} detenido.")

    def get_dominant_color(self, image_crop):
        # Asegurarse de que la imagen no esté vacía
        if image_crop.size == 0:
            return None
        # Convertir a RGB y remodelar para KMeans
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pixels = image_crop.reshape((-1, 3))
        
        # Usar KMeans para encontrar el color dominante
        # n_clusters=3 para obtener un color representativo, ignorando sombras/brillos extremos
        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(pixels)
        # El color dominante es el centro del clúster más grande
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_color_rgb = kmeans.cluster_centers_[unique[np.argmax(counts)]]
        return '#%02x%02x%02x' % (int(dominant_color_rgb[0]), int(dominant_color_rgb[1]), int(dominant_color_rgb[2]))

    def save_event_if_needed(self, frame, results):
        # Condición: Hay movimiento, hay detecciones y ha pasado el cooldown
        current_time = time.time()
        if not self.motion_detected or not results.boxes.cls.numel() > 0 or (current_time - self.last_event_time < self.event_cooldown):
            return

        self.last_event_time = current_time
        timestamp = datetime.now()
        
        # Crear nombres de archivo
        base_filename = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{base_filename}.jpg"
        json_filename = f"{base_filename}.json"
        
        image_path = os.path.join(self.output_dir, image_filename)
        json_path = os.path.join(self.output_dir, json_filename)
        
        # Guardar la imagen
        cv2.imwrite(image_path, frame)
        print(f"Evento guardado: {image_path}")

        # Preparar datos para el JSON
        objects_list = []
        objects_count = {}
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = self.coco_classes[class_id]
            bbox = [int(coord) for coord in box.xyxy[0]] # [x1, y1, x2, y2]

            # Recortar el objeto de la imagen original
            x1, y1, x2, y2 = bbox
            object_crop = frame[y1:y2, x1:x2]

            object_data = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf),
                "bbox": bbox
            }

            # --- Lógica de análisis de apariencia ---
            if class_name == 'person':
                mid_y = y1 + (y2 - y1) // 2
                upper_body_bbox = [x1, y1, x2, mid_y]
                lower_body_bbox = [x1, mid_y, x2, y2]

                upper_crop = frame[y1:mid_y, x1:x2]
                lower_crop = frame[mid_y:y2, x1:x2]

                object_data['appearance'] = {
                    "upper_body": {
                        "bbox": upper_body_bbox,
                        "dominant_color_hex": self.get_dominant_color(upper_crop)
                    },
                    "lower_body": {
                        "bbox": lower_body_bbox,
                        "dominant_color_hex": self.get_dominant_color(lower_crop)
                    }
                }
            else: # Para otros objetos, obtener el color general
                object_data['dominant_color_hex'] = self.get_dominant_color(object_crop)

            objects_list.append(object_data)
            objects_count[class_name] = objects_count.get(class_name, 0) + 1

        event_data = {
            "timestamp_utc": timestamp.isoformat(),
            "camera_id": self.id,
            "camera_name": self.name,
            "image_path": os.path.join(str(self.id), image_filename), # Ruta relativa para la web
            "objects": objects_list,
            "objects_count": objects_count
        }

        # Guardar el archivo JSON
        with open(json_path, 'w') as f:
            json.dump(event_data, f, indent=2)

    def stop(self):
        self.is_running = False

    def get_frame(self):
        with self.lock:
            return self.last_frame

    def update_config(self, key, value):
        with self.lock:
            if key == 'show_bbox':
                self.show_bbox = value
                return # No es necesario modificar el config dict para esto
            if key == 'show_motion':
                self.show_motion_contours = value
                return
            if key == 'motion_sensitivity':
                self.motion_sensitivity = int(value)
                self.config[key] = int(value) # Guardar en config para consistencia
                return

            self.config[key] = value


camera_threads = {}

def load_config():
    global camera_threads
    # Detener hilos antiguos si existen
    for thread in camera_threads.values():
        thread.stop()
        thread.join()
    camera_threads.clear()

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            for cam_config in config.get('cameras', []):
                thread = CameraThread(cam_config)
                thread.start()
                camera_threads[cam_config['name']] = thread
            print("Configuración cargada. Cámaras activas:", list(camera_threads.keys()))
    except Exception as e:
        print(f"Error al cargar config.yml: {e}")
        camera_threads = {}

# --- Generación de Frames de Video ---
def generate_frames(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        print(f"Error: Cámara '{camera_name}' no encontrada en la configuración.")
        return

    while True:
        time.sleep(0.03) # Limita la tasa de frames para no saturar
        frame_bytes = thread.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Rutas de la Aplicación ---
@app.route('/')
def index():
    # Pasa la configuración completa de las cámaras a la plantilla
    camera_configs = [t.config for t in camera_threads.values()]
    return render_template('index.html', cameras=camera_configs)

@app.route('/api/status')
def api_status():
    status = {}
    for name, thread in camera_threads.items():
        status[name] = {
            "enabled": thread.config.get('enabled', False),
            "detected_classes": [int(c) for c in thread.detected_classes_in_frame], # Corregido: convertir a int
            "motion_detected": thread.motion_detected
        }
    return jsonify(status)

@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    return Response(generate_frames(camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/<camera_name>/toggle', methods=['POST'])
def toggle_camera(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404
    
    is_enabled = request.json.get('enabled')
    thread.update_config('enabled', is_enabled)
    return jsonify({"status": "success", "message": f"Cámara {camera_name} {'activada' if is_enabled else 'desactivada'}"})

@app.route('/api/camera/<camera_name>/toggle_bbox', methods=['POST'])
def toggle_bbox(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404
    
    show = request.json.get('show')
    thread.update_config('show_bbox', show)
    return jsonify({"status": "success", "message": f"Bounding boxes {'visibles' if show else 'ocultos'} para {camera_name}"})

@app.route('/api/camera/<camera_name>/toggle_motion', methods=['POST'])
def toggle_motion(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404
    
    show = request.json.get('show')
    thread.update_config('show_motion', show)
    return jsonify({"status": "success", "message": f"Visualización de movimiento {'activada' if show else 'desactivada'} para {camera_name}"})

@app.route('/api/camera/<camera_name>/classes', methods=['POST'])
def update_classes(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404

    classes = request.json.get('classes', [])
    # Convertir a enteros
    classes = [int(c) for c in classes]
    thread.update_config('detect_classes', classes)
    return jsonify({"status": "success", "message": f"Clases de detección actualizadas para {camera_name}"})

@app.route('/api/camera/<camera_name>/motion_sensitivity', methods=['POST'])
def update_motion_sensitivity(camera_name):
    thread = camera_threads.get(camera_name)
    if not thread:
        return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404

    sensitivity = request.json.get('sensitivity')
    if sensitivity is None:
        return jsonify({"status": "error", "message": "Sensibilidad no proporcionada"}), 400
    thread.update_config('motion_sensitivity', sensitivity)
    return jsonify({"status": "success", "message": f"Sensibilidad de movimiento actualizada para {camera_name}"})

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        try:
            with open(CONFIG_PATH, 'r') as f:
                content = f.read()
            return Response(content, mimetype='text/yaml')
        except FileNotFoundError:
            return "config.yml no encontrado", 404
    
    if request.method == 'POST':
        new_content = request.json.get('content')
        with open(CONFIG_PATH, 'w') as f:
            f.write(new_content)
        # No recargamos aquí para no interrumpir los streams.
        # El usuario debe refrescar la página para aplicar cambios mayores.
        return jsonify({"status": "success", "message": "Configuración guardada. Refresca la página para ver los cambios."})

@app.route('/api/event_cameras')
def get_event_cameras():
    try:
        # Escanea el directorio 'static' para encontrar carpetas de cámaras (que son los IDs)
        camera_ids = [d for d in os.listdir(STATIC_PATH) if os.path.isdir(os.path.join(STATIC_PATH, d))]
        return jsonify(sorted(camera_ids))
    except Exception as e:
        print(f"Error al listar cámaras con eventos: {e}")
        return jsonify([])

@app.route('/api/events')
def get_events():
    camera_id = request.args.get('camera_id')
    date_str = request.args.get('date') # Formato YYYY-MM-DD

    if not camera_id:
        return jsonify({"error": "camera_id es requerido"}), 400

    events = []
    cam_dir = os.path.join(STATIC_PATH, camera_id)
    if not os.path.isdir(cam_dir):
        return jsonify([])

    # Formato de fecha para filtrar nombres de archivo: YYYYMMDD
    file_date_prefix = date_str.replace('-', '') if date_str else ''

    for filename in sorted(os.listdir(cam_dir), reverse=True):
        if filename.endswith('.json') and filename.startswith(file_date_prefix):
            try:
                with open(os.path.join(cam_dir, filename), 'r') as f:
                    events.append(json.load(f))
            except Exception as e:
                print(f"Error al leer el archivo de evento {filename}: {e}")
    return jsonify(events)

if __name__ == '__main__':
    load_config() # Carga inicial de la configuración
    # debug=False es importante cuando se usan hilos para evitar que Flask inicie la app dos veces
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
