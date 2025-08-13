import os

class Config:
    SECRET_KEY = 'your-secret-key-here'  # Can be any random string
    SQLALCHEMY_DATABASE_URI = 'sqlite:///polyview.db'
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
    
    # Create uploads directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)