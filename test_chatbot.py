from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_create_chatbot():
    with open("test.pdf", "rb") as pdf_file:
        response = client.post("/chatbot/create/", files={"pdf": pdf_file}, json={"prompt": "test prompt"})
        assert response.status_code == 200
        assert response.json()["message"] == "Chatbot created successfully"

def test_chat():
    response = client.post("/chatbot/chat/", json={"query": "What is this PDF about?"})
    assert response.status_code == 200
    assert "answer" in response.json()
