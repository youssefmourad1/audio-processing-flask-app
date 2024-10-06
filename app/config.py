import os

class Config:
    GROQ_KEY = os.getenv('GROQ_KEY', 'your_default_groq_key')
    HG_KEY = os.getenv('HG_KEY', 'your_default_hg_key')
    UPLOAD_FOLDER = 'uploads/'
    OUTPUT_FOLDER = 'outputs/'
    ALLOWED_EXTENSIONS = {'wav', 'mp3'}
