from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
import sqlite3
from models import User, LoginUser
from database import create_user_table, create_connection

auth_router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Ensure the user table exists
create_user_table()

@auth_router.post("/register/")
async def register(user: User):
    hashed_password = pwd_context.hash(user.password)
    with create_connection() as conn:
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username already registered")
    return {"message": "User registered successfully"}

@auth_router.post("/login/")
async def login(user: LoginUser):
    conn = create_connection()
    cursor = conn.execute("SELECT password FROM users WHERE username=?", (user.username,))
    record = cursor.fetchone()
    if record and pwd_context.verify(user.password, record[0]):
        return {"message": "Login successful"}
    raise HTTPException(status_code=400, detail="Invalid username or password")
