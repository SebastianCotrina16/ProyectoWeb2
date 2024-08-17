"""
Creates a SQLite database with a table to store the users' information.
"""
import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE users (
        dni TEXT PRIMARY KEY,
        name TEXT,
        face_id INTEGER
    )
''')
conn.commit()
conn.close()
