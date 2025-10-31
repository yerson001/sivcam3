import os
import yaml
import json
import time
from functools import wraps
import queue
from flask import Flask, Response, render_template, request, jsonify, Blueprint
from ultralytics import YOLO

# Importar nuestras clases personalizadas
from camera import CameraThread
from synchronizer import SyncThread

# --- Variables Globales de Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yml')
STATIC_PATH = os.path.join(BASE_DIR, 'static')


class VideoAnalysisApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_paths()
        
        print("Cargando modelo YOLOv8...")
        self.yolo_model = YOLO("yolov8n.pt")
        print("Modelo cargado.")
        
        self.camera_threads = {}
        self.event_sync_queue = queue.Queue()
        # Inyectar la cola y una función para obtener los hilos de cámara
        self.sync_thread = SyncThread(self.event_sync_queue, lambda: self.camera_threads)
        
        self.register_blueprints()
        self.register_routes()
        self.load_config_and_start_threads()

    def setup_paths(self):
        # Asegurarse de que el directorio estático exista
        os.makedirs(STATIC_PATH, exist_ok=True)

    def register_blueprints(self):
        assets_bp = Blueprint('assets', __name__, static_folder='assets', static_url_path='/assets')
        self.app.register_blueprint(assets_bp)

    def _load_config_from_file(self):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print("Advertencia: config.yml no encontrado. Creando uno vacío.")
            return {}
        except Exception as e:
            print(f"Error al cargar config.yml: {e}")
            return {}

    def _save_config_to_file(self, config_data):
        try:
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error al guardar config.yml: {e}")
            return False

    def load_config_and_start_threads(self):
        print("Cargando configuración y reiniciando hilos...")
        # Detener hilos antiguos si existen
        for thread in self.camera_threads.values():
            thread.stop()
            thread.join()
        self.camera_threads.clear()

        config = self._load_config_from_file()
        
        # Iniciar hilos de cámara
        for cam_config in config.get('cameras', []):
            try:
                thread = CameraThread(cam_config, self.yolo_model, STATIC_PATH, self.event_sync_queue)
                thread.start()
                self.camera_threads[cam_config['name']] = thread
            except Exception as e:
                print(f"Error al iniciar la cámara {cam_config.get('name')}: {e}")
                
        print("Cámaras activas:", list(self.camera_threads.keys()))
        
        # Actualizar configuración de sincronización
        sync_config = config.get('sync', {})
        self.sync_thread.update_config(sync_config)

    def get_camera_thread_decorator(self, f):
        """Decorador para obtener el hilo de la cámara y manejar errores."""
        @wraps(f)
        def decorated_function(camera_name, *args, **kwargs):
            thread = self.camera_threads.get(camera_name)
            if not thread:
                return jsonify({"status": "error", "message": "Cámara no encontrada"}), 404
            return f(thread, *args, **kwargs)
        return decorated_function

    def generate_frames(self, camera_name):
        """Generador de frames para el stream de video."""
        thread = self.camera_threads.get(camera_name)
        if not thread:
            print(f"Error: Cámara '{camera_name}' no encontrada.")
            return

        while True:
            time.sleep(0.03) # Limitar tasa de frames
            frame_bytes = thread.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def register_routes(self):
        """Define todas las rutas de Flask."""
        
        @self.app.route('/')
        def index():
            camera_configs = [t.config for t in self.camera_threads.values()]
            sync_config = self.sync_thread.sync_config
            return render_template('index.html', cameras=camera_configs, sync_config=sync_config)

        @self.app.route('/video_feed/<camera_name>')
        def video_feed(camera_name):
            return Response(self.generate_frames(camera_name),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # --- API de Estado ---
        @self.app.route('/api/status')
        def api_status():
            status = {}
            for name, thread in self.camera_threads.items():
                status[name] = {
                    "enabled": thread.config.get('enabled', False),
                    "detected_classes": [int(c) for c in thread.detected_classes_in_frame],
                    "motion_detected": thread.motion_detected
                }
            return jsonify(status)

        # --- API de Control de Cámaras ---
        @self.app.route('/api/camera/<camera_name>/toggle', methods=['POST'])
        @self.get_camera_thread_decorator
        def toggle_camera(thread):
            is_enabled = request.json.get('enabled')
            thread.update_config('enabled', is_enabled)
            return jsonify({"status": "success", "message": f"Cámara {thread.name} {'activada' if is_enabled else 'desactivada'}"})

        @self.app.route('/api/camera/<camera_name>/toggle_bbox', methods=['POST'])
        @self.get_camera_thread_decorator
        def toggle_bbox(thread):
            show = request.json.get('show')
            thread.update_config('show_bbox', show)
            return jsonify({"status": "success"})

        @self.app.route('/api/camera/<camera_name>/toggle_motion', methods=['POST'])
        @self.get_camera_thread_decorator
        def toggle_motion(thread):
            show = request.json.get('show')
            thread.update_config('show_motion', show)
            return jsonify({"status": "success"})

        @self.app.route('/api/camera/<camera_name>/classes', methods=['POST'])
        @self.get_camera_thread_decorator
        def update_classes(thread):
            classes = [int(c) for c in request.json.get('classes', [])]
            thread.update_config('detect_classes', classes)
            return jsonify({"status": "success"})

        @self.app.route('/api/camera/<camera_name>/motion_sensitivity', methods=['POST'])
        @self.get_camera_thread_decorator
        def update_motion_sensitivity(thread):
            sensitivity = request.json.get('sensitivity')
            if sensitivity is None:
                return jsonify({"status": "error", "message": "Sensibilidad no proporcionada"}), 400
            thread.update_config('motion_sensitivity', sensitivity)
            return jsonify({"status": "success"})

        # --- API de Configuración General ---
        @self.app.route('/api/config', methods=['GET', 'POST'])
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
                try:
                    # Validar YAML antes de guardar
                    yaml.safe_load(new_content) 
                    with open(CONFIG_PATH, 'w') as f:
                        f.write(new_content)
                    
                    # --- Lógica de recarga inteligente ---
                    # Obtenemos la configuración de cámaras antes y después de guardar.
                    old_cam_configs = [t.config for t in self.camera_threads.values()]
                    new_config = yaml.safe_load(new_content)
                    new_cam_configs = new_config.get('cameras', [])
                    
                    # Solo reiniciamos los hilos si la sección 'cameras' ha cambiado.
                    if old_cam_configs != new_cam_configs:
                        self.load_config_and_start_threads()
                    return jsonify({"status": "success", "message": "Configuración guardada y aplicada."})
                except yaml.YAMLError as e:
                    return jsonify({"status": "error", "message": f"Error en el formato YAML: {e}"}), 400
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Error al guardar: {e}"}), 500

        # --- API de Sincronización ---
        @self.app.route('/api/sync_config', methods=['GET', 'POST'])
        def handle_sync_config():
            if request.method == 'GET':
                return jsonify(self.sync_thread.sync_config)
            
            if request.method == 'POST':
                new_sync_config = request.json
                self.sync_thread.update_config(new_sync_config)
                
                full_config = self._load_config_from_file()
                full_config['sync'] = new_sync_config
                
                if self._save_config_to_file(full_config):
                    return jsonify({"status": "success", "message": "Configuración de sincronización guardada."})
                else:
                    return jsonify({"status": "error", "message": "Error al guardar en config.yml"}), 500

        # --- API de Eventos ---
        @self.app.route('/api/event_cameras')
        def get_event_cameras():
            try:
                camera_ids = [d for d in os.listdir(STATIC_PATH) if os.path.isdir(os.path.join(STATIC_PATH, d))]
                return jsonify(sorted(camera_ids))
            except Exception as e:
                print(f"Error al listar cámaras con eventos: {e}")
                return jsonify([])

        @self.app.route('/api/events')
        def get_events():
            camera_id = request.args.get('camera_id')
            date_str = request.args.get('date') # YYYY-MM-DD
            if not camera_id:
                return jsonify({"error": "camera_id es requerido"}), 400

            events = []
            cam_dir = os.path.join(STATIC_PATH, camera_id)
            if not os.path.isdir(cam_dir):
                return jsonify([])

            file_date_prefix = date_str.replace('-', '') if date_str else ''
            
            try:
                filenames = sorted(os.listdir(cam_dir), reverse=True)
                for filename in filenames:
                    if filename.endswith('.json') and filename.startswith(file_date_prefix):
                        try:
                            with open(os.path.join(cam_dir, filename), 'r') as f:
                                events.append(json.load(f))
                        except Exception as e:
                            print(f"Error al leer evento {filename}: {e}")
            except Exception as e:
                 print(f"Error al listar directorio {cam_dir}: {e}")

            return jsonify(events)

    def run(self):
        """Inicia el hilo de sincronización y la aplicación Flask."""
        try:
            self.sync_thread.start()
            print("Iniciando servidor Flask en http://0.0.0.0:5000")
            # debug=False es crucial para evitar que los hilos se inicien dos veces
            self.app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        except KeyboardInterrupt:
            print("Deteniendo la aplicación...")
        finally:
            # Asegurarse de que todos los hilos se detengan al salir
            for thread in self.camera_threads.values():
                thread.stop()
            self.sync_thread.stop()
            print("Aplicación detenida.")


if __name__ == '__main__':
    main_app = VideoAnalysisApp()
    main_app.run()