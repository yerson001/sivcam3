import threading
import time
import requests
from datetime import datetime, timedelta
import queue

class SyncThread(threading.Thread):
    def __init__(self, event_sync_queue):
        super().__init__()
        self.daemon = True
        self.is_running = True
        self.sync_config = {}
        self.lock = threading.Lock()
        self.event_sync_queue = event_sync_queue
        
        print("Inicializando hilo de Sincronización.")

    def run(self):
        """
        Espera eventos en la cola y los procesa uno por uno.
        La lógica de filtrado de eventos ya se ha hecho en CameraThread.
        Este hilo solo se encarga de consumir la cola y enviar.
        """
        while self.is_running:
            try:
                # Espera bloqueante hasta que haya un item en la cola.
                event_data = self.event_sync_queue.get(timeout=1)
                self.send_event(event_data)
                self.event_sync_queue.task_done()
            except queue.Empty:
                # La cola está vacía, simplemente continuamos el bucle.
                continue
            except Exception as e:
                print(f"Sync Error en el bucle principal: {e}")

    def send_event(self, event_data):
        with self.lock:
            is_enabled = self.sync_config.get('enabled', False)
            endpoint = self.sync_config.get('endpoint')
            cameras_to_sync = self.sync_config.get('cameras', [])
            auth_token = self.sync_config.get('auth_token')

        cam_name = event_data['camera_name']

        # Verificar si la sincronización está habilitada y si esta cámara debe sincronizarse
        if not is_enabled or not endpoint:
            return
        if "all" not in cameras_to_sync and cam_name not in cameras_to_sync:
            print(f"Sync: Evento de '{cam_name}' ignorado (no está en la lista de sincronización).")
            return

        print(f"Sync: Procesando evento de '{cam_name}' para enviar al backend.")

        frame_bytes = event_data.get('frame_bytes')
        if frame_bytes:
            try:
                # Preparamos los datos para que coincidan con tu API.
                files = [('image', ('frame.jpg', frame_bytes, 'image/jpeg'))]
                payload = {
                    'cameraId': event_data['camera_id'],
                    'videoName': cam_name
                }

                headers = {}
                if auth_token:
                    headers['Authorization'] = auth_token

                try:
                    response = requests.post(endpoint, files=files, data=payload, headers=headers, timeout=20)
                    print(f"Sync: Frame de '{cam_name}' enviado. Status: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Sync Error: No se pudo enviar el frame de '{cam_name}' a {endpoint}. Error: {e}")
            except Exception as e:
                print(f"Sync Error: Fallo inesperado al procesar el evento de '{cam_name}'. Error: {e}")
    def update_config(self, new_config):
        with self.lock:
            old_enabled = self.sync_config.get('enabled', False)
            new_enabled = new_config.get('enabled', False)

            # Mostrar un mensaje en la terminal si el estado de activación cambia.
            if new_enabled and not old_enabled:
                print("\n✅ Sincronización ACTIVADA. Esperando eventos para enviar al backend.\n")
            elif not new_enabled and old_enabled:
                print("\n❌ Sincronización DESACTIVADA.\n")

            self.sync_config = new_config

    def stop(self):
        self.is_running = False