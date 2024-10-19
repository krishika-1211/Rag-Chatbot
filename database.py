import sqlite3
from typing import List, Dict

def create_connection():
    conn = sqlite3.connect("users.db")
    return conn

def create_user_table():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.close()

def create_chatbot_table():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS chatbots
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE, description TEXT, prompt TEXT)''')
    conn.close()

def add_chatbot(name: str, description: str, prompt: str) -> int:
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO chatbots (name, description, prompt) VALUES (?, ?, ?)''', 
                   (name, description, prompt))
    conn.commit()
    chatbot_id = cursor.lastrowid
    conn.close()
    return chatbot_id

async def get_top_chatbots(limit: int = 5) -> List[Dict]:
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM chatbots LIMIT ?''', (limit,))
    chatbots = cursor.fetchall()
    conn.close()
    return [{"id": id, "name": name, "description": description, "prompt": prompt} 
            for id, name, description, prompt in chatbots]

# Call this function at startup to ensure the tables exist
def initialize_database():
    create_user_table()
    create_chatbot_table()
