from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import auth_router
from chatbot import chatbot_router
from database import initialize_database

app = FastAPI()

initialize_database()

# CORS settings
origins = ["*"]  # origins that are allowed to make requests to the api

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Chatbot API"}
