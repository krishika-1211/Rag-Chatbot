from pydantic import BaseModel, Field
from fastapi import Form

# User model for registration and login
class User(BaseModel):
    username: str = Field(..., min_length=1, example="username")
    password: str = Field(..., min_length=8, example="password")

class LoginUser(BaseModel):
    username: str = Field(..., min_length=1, example="username")
    password: str = Field(..., min_length=8, example="password")

# Chatbot creation
class ChatbotRequest(BaseModel):
    prompt: str

# Chat query
class ChatQuery(BaseModel):
    query: str = Form(...)

class ChatbotRequest(BaseModel):
    name: str
    description: str
    prompt: str
