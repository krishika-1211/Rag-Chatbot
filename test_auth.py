from fastapi.testclient import TestClient
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def test_register():
    response = client.post("/auth/register/", json={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    assert response.json() == {"message": "User registered successfully"}

def test_login():
    response = client.post("/auth/login/", json={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    assert response.json() == {"message": "Login successful"}
