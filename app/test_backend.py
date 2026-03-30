"""
app/test_backend.py
-------------------
Quick integration test to verify the FastAPI /predict endpoint.
"""

from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_predict_endpoint():
    print("Testing POST /predict...")
    
    payload = {
        "session_id": "test-session-001",
        "telemetry": {
            "deaths": 5,
            "retries": 4,
            "score_delta": -20.0,
            "streak": -3,
            "reaction_time_ms": 450,
            "input_speed": 2.1
        },
        "nlp": {
            "text": "this is impossible",
            "intent_gap": 0.6
        }
    }
    
    response = client.post("/api/v1/predict", json=payload)
    
    if response.status_code == 200:
        print("[SUCCESS] Status 200 OK")
        print("\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"[ERROR] Status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_predict_endpoint()
