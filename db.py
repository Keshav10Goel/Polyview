# import sqlite3

# def init_db():
#     conn = sqlite3.connect('polyview.db')
#     c = conn.cursor()
    
#     # Create users table
#     c.execute('''CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         email TEXT UNIQUE NOT NULL,
#         password TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     )''')
    
#     # Create analysis table
#     c.execute('''CREATE TABLE IF NOT EXISTS analyses (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER NOT NULL,
#         truth_score REAL NOT NULL,
#         transcript TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#         FOREIGN KEY(user_id) REFERENCES users(id)
#     )''')
    
#     conn.commit()
#     conn.close()

# if __name__ == '__main__':
#     init_db()
#     print("Database initialized successfully")


import sqlite3

def init_db():
    conn = sqlite3.connect('polyview.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Create analyses table
    c.execute('''CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        truth_score REAL NOT NULL,
        transcript TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    # Insert sample user
    try:
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                 ("Test User", "test@example.com", "test123"))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

if __name__ == '__main__':
    init_db()