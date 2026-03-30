"""
game/api_client.py
------------------
Async HTTP client for Pygame. 
Spins a background thread to send telemetry and receive difficulty 
adjustments without dropping game FPS.
"""

import threading
import queue
import requests

API_URL = "http://localhost:8001/predict"

class GameAPIClient:
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Thread communication queues
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def send_telemetry(self, telemetry_payload: dict):
        """Asynchronously dispatch data to the new lightweight API."""
        full_payload = {
            "telemetry": telemetry_payload.get("telemetry", {}),
            "nlp": telemetry_payload.get("nlp", {})
        }
        self.request_queue.put(full_payload)
        
    def check_for_updates(self) -> dict | None:
        """
        Check if the backend has returned a new difficulty/state.
        Returns None if nothing is ready yet.
        """
        try:
            # Non-blocking get
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self):
        """Background thread loop that blocks on queue, makes request, then queues response."""
        while True:
            payload = self.request_queue.get() # blocks until payload item is put in
            
            try:
                # 2 second timeout - if backend takes longer, drop the request
                response = requests.post(API_URL, json=payload, timeout=2.0)
                
                if response.status_code == 200:
                    data = response.json()
                    self.response_queue.put({
                        "cii": data["cii"],
                        "state": data["state"],
                        "action": data["action"],
                        "confidence": data.get("confidence", 0.0)
                    })
                else:
                    print(f"[API ERROR] Status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"[API CONNECTION ERROR] {e}")
            except Exception as e:
                print(f"[API CLIENT CRASH] {e} -> Data parsed: {data if 'data' in locals() else 'None'}")
                
            finally:
                # Mark as done so queue.join() could theoretically work
                self.request_queue.task_done()
