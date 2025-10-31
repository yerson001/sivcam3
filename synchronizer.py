import threading
import time
import requests
from datetime import datetime
import queue

class SyncThread(threading.Thread):
    def __init__(self, event_sync_queue, get_camera_threads_callback):
        super().__init__()
        self.daemon = True
        self.is_running = True
        self.sync_config = {
            'enabled': True,        # Sincronización activa o no
            'sync_mode': 'event',     # 'event', 'periodic', o 'both'
            'interval_seconds': 30,   # Intervalo para el modo periódico
            'test_mode': True,       # Modo prueba (no envía realmente)
            'endpoint': "http://34.144.231.224/api/video-data/analyze-frame",        # URL del servidor
            'cameras': ['all'],      # Lista de cámaras a sincronizar
            'auth_token': None       # Token de autenticación opcional
        }
        self.lock = threading.Lock()
        self.event_sync_queue = event_sync_queue
        self.get_camera_threads = get_camera_threads_callback
        self.last_periodic_send_time = 0
        print("Inicializando hilo de Sincronización.")

    def run(self):
        """Consume eventos de la cola y los envía al servidor."""
        while self.is_running:
            try:
                event_data = self.event_sync_queue.get(timeout=1)
                
                with self.lock:
                    mode = self.sync_config.get('sync_mode', 'event')
                
                if mode in ['event', 'both']:
                    self.send_event(event_data)
                    self.event_sync_queue.task_done()

            except queue.Empty:
                # La cola está vacía, no hacemos nada con eventos.
                # Pasamos a la lógica periódica.
                pass
            except Exception as e:
                print(f"Sync Error en el bucle principal: {e}")

            # --- Lógica de envío periódico ---
            with self.lock:
                mode = self.sync_config.get('sync_mode', 'event')
                interval = self.sync_config.get('interval_seconds', 30)
            
            if mode in ['periodic', 'both']:
                current_time = time.time()
                if (current_time - self.last_periodic_send_time) > interval:
                    print(f"Sync: Disparando envío periódico (cada {interval}s).")
                    self.send_periodic_frames()
                    self.last_periodic_send_time = current_time

    def send_periodic_frames(self):
        """Obtiene el frame actual de las cámaras y lo envía."""
        camera_threads = self.get_camera_threads()
        if not camera_threads:
            return

        for cam_name, cam_thread in camera_threads.items():
            frame_bytes = cam_thread.get_frame()
            if frame_bytes:
                # Construimos un payload similar al de un evento
                payload = {
                    "camera_name": cam_thread.name,
                    "camera_id": cam_thread.id,
                    "frame_bytes": frame_bytes
                }
                # Usamos un hilo para no bloquear el bucle si un envío tarda
                threading.Thread(target=self.send_event, args=(payload,)).start()

    def send_event(self, event_data):
        with self.lock:
            is_enabled = self.sync_config.get('enabled', False)
            test_mode = self.sync_config.get('test_mode', True)
            endpoint = self.sync_config.get('endpoint')
            cameras_to_sync = self.sync_config.get('cameras', [])
            auth_token = self.sync_config.get('auth_token')

        cam_name = event_data.get('camera_name')

        # Validaciones básicas
        if not is_enabled or not endpoint:
            return
        if "all" not in cameras_to_sync and cam_name not in cameras_to_sync:
            print(f"Sync: Evento de '{cam_name}' ignorado (no está en la lista de sincronización).")
            return

        frame_bytes = event_data.get('frame_bytes')
        if not frame_bytes:
            print(f"Sync: No hay frame para enviar desde '{cam_name}'.")
            return

        # Modo prueba → solo simula envío
        if test_mode:
            print(f"\n[MODO PRUEBA] Simulando envío de frame de '{cam_name}' a {endpoint}")
            print(f"  Camera ID: {event_data.get('camera_id')}, Bytes del frame: {len(frame_bytes)}\n")
            time.sleep(0.5)
            return

        # Modo real → enviar al servidor
        try:
            files = [('image', ('frame.jpg', frame_bytes, 'image/jpeg'))]
            payload = {
                'cameraId': event_data.get('camera_id'),
                'videoName': cam_name
            }
            headers = {}
            if auth_token:
                headers['Authorization'] = auth_token

            print(f"Sync: Enviando frame de '{cam_name}' a {endpoint}...")
            response = requests.post(endpoint, files=files, data=payload, headers=headers, timeout=20)
            print(f"Sync: Frame de '{cam_name}' enviado. Status: {response.status_code}")
            try:
                print(response.json())  # Mostrar JSON de respuesta si existe
            except:
                print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Sync Error: No se pudo enviar el frame de '{cam_name}'. Error: {e}")
        except Exception as e:
            print(f"Sync Error inesperado al procesar '{cam_name}': {e}")

    def update_config(self, new_config):
        with self.lock:
            old_enabled = self.sync_config.get('enabled', False)
            new_enabled = new_config.get('enabled', False)
            old_test_mode = self.sync_config.get('test_mode', True)
            new_test_mode = new_config.get('test_mode', True)
            old_mode = self.sync_config.get('sync_mode', 'event')
            new_mode = new_config.get('sync_mode', 'event')

            if new_enabled != old_enabled:
                if new_enabled:
                    print("\n✅ Sincronización ACTIVADA.\n")
                else:
                    print("\n❌ Sincronización DESACTIVADA.\n")

            if new_test_mode != old_test_mode:
                if new_test_mode:
                    print("\n🧪 MODO PRUEBA ACTIVADO.\n")
                else:
                    print("\n🚀 MODO REAL ACTIVADO.\n")

            if new_mode != old_mode:
                print(f"\n🔄 MODO DE SINCRONIZACIÓN CAMBIADO A: '{new_mode.upper()}'\n")

            self.sync_config.update(new_config)

    def stop(self):
        self.is_running = False
        print("Hilo de sincronización detenido.")
