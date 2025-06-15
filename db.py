import sqlite3
from datetime import datetime
import os

DB_PATH = "data/chatgpt_clone.sqlite"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        role TEXT,
        type TEXT,
        content TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )''')
    conn.commit()
    conn.close()

def new_chat(name="New Chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (name, created_at) VALUES (?,?)", (name, datetime.now()))
    conn.commit()
    chat_id = c.lastrowid
    conn.close()
    return chat_id

def get_chats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM chats ORDER BY created_at DESC")
    chats = c.fetchall()
    conn.close()
    return chats

def get_messages(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, type, content FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,))
    messages = c.fetchall()
    conn.close()
    return [{"role": r, "type": t, "content": c} for r, t, c in messages]

def add_message(chat_id, role, type_, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (chat_id, role, type, content, created_at) VALUES (?, ?, ?, ?, ?)",
              (chat_id, role, type_, content, datetime.now()))
    conn.commit()
    conn.close()
