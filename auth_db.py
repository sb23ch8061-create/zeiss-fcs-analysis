import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path

# Define database path
DB_FILE = Path("fcs_app.db")

def init_db():
    """Initializes the SQLite database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: bytes) -> str:
    """Hashes a password with a given salt securely using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt, 
        100000
    ).hex()

def create_user(username, password):
    """Creates a new user. Returns (True, "Success message") or (False, "Error message")."""
    if not username or not password:
        return False, "Username and password cannot be empty."
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "Username already exists. Please choose another."
        
    # Generate secure random salt and hash the password
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
    """Verifies a user's login credentials. Returns True if valid, False otherwise."""
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    record = cursor.fetchone()
    conn.close()
    
    if record is None:
        return False  # User not found
        
    stored_hash, stored_salt_hex = record
    stored_salt = bytes.fromhex(stored_salt_hex)
    
    # Hash the provided password with the stored salt to see if they match
    attempt_hash = _hash_password(password, stored_salt)
    
    return attempt_hash == stored_hash
    
# Automatically initialize the database when this module is imported
init_db()