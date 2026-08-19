import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path

# Define database path
DB_FILE = Path("fcs_app.db")

def init_db():
    """Initializes the SQLite database and creates necessary tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Projects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 3. Datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# --- AUTHENTICATION FUNCTIONS ---

def _hash_password(password: str, salt: bytes) -> str:
    """Hashes a password securely."""
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()

def create_user(username, password):
    """Creates a new user."""
    if not username or not password:
        return False, "Username and password cannot be empty."
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "Username already exists. Please choose another."
        
    salt = os.urandom(32)
    password_hash = _hash_password(password, salt)
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt.hex(), datetime.now())
        )
        conn.commit()
        success = True
        msg = "Account created successfully. You can now log in."
    except Exception as e:
        success = False
        msg = f"Database error: {e}"
    finally:
        conn.close()
        
    return success, msg

def verify_user(username, password):
    """Verifies user login."""
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    record = cursor.fetchone()
    conn.close()
    
    if record is None:
        return False
        
    stored_hash, stored_salt_hex = record
    attempt_hash = _hash_password(password, bytes.fromhex(stored_salt_hex))
    return attempt_hash == stored_hash

def get_user_id(username):
    """Retrieves the internal database ID for a given username."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    record = cursor.fetchone()
    conn.close()
    return record[0] if record else None

# --- PROJECT MANAGEMENT FUNCTIONS ---

def create_project(username, project_name):
    """Creates a new project for the specified user."""
    user_id = get_user_id(username)
    if not user_id:
        return False, "User not found."
        
    if not project_name.strip():
        return False, "Project name cannot be empty."
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM projects WHERE user_id = ? AND name = ?", (user_id, project_name))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "A project with this name already exists."
        
    try:
        cursor.execute("INSERT INTO projects (user_id, name) VALUES (?, ?)", (user_id, project_name))
        conn.commit()
        success, msg = True, "Project created successfully."
    except Exception as e:
        success, msg = False, f"Error creating project: {e}"
    finally:
        conn.close()
    return success, msg

def get_user_projects(username):
    """Returns a list of project names and IDs for a user."""
    user_id = get_user_id(username)
    if not user_id:
        return []
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM projects WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    projects = cursor.fetchall()
    conn.close()
    
    return [{"id": p[0], "name": p[1], "created_at": p[2]} for p in projects]

# --- DATASET MANAGEMENT FUNCTIONS (NEW) ---

def get_project_id(username, project_name):
    """Retrieves the internal database ID for a specific project."""
    user_id = get_user_id(username)
    if not user_id: 
        return None
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM projects WHERE user_id = ? AND name = ?", (user_id, project_name))
    record = cursor.fetchone()
    conn.close()
    
    return record[0] if record else None

def save_dataset_record(username, project_name, filename, file_path):
    """Saves a record of an uploaded dataset to the database."""
    project_id = get_project_id(username, project_name)
    if not project_id: 
        return False, "Project not found."
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if file already exists in db for this project
        cursor.execute("SELECT id FROM datasets WHERE project_id = ? AND filename = ?", (project_id, filename))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO datasets (project_id, filename, file_path) VALUES (?, ?, ?)",
                (project_id, filename, str(file_path))
            )
            conn.commit()
        success, msg = True, "Dataset recorded successfully."
    except Exception as e:
        success, msg = False, f"Database error: {e}"
    finally:
        conn.close()
        
    return success, msg

def get_project_datasets(username, project_name):
    """Retrieves all datasets associated with a specific project."""
    project_id = get_project_id(username, project_name)
    if not project_id: 
        return []
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, file_path, upload_timestamp FROM datasets WHERE project_id = ? ORDER BY upload_timestamp DESC", (project_id,))
    records = cursor.fetchall()
    conn.close()
    
    return [{"id": r[0], "filename": r[1], "file_path": r[2], "upload_timestamp": r[3]} for r in records]

# Automatically initialize the database when imported
init_db()