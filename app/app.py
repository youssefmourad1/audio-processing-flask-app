from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from .helpers import process_audio

app = Flask(__name__)
app.config.from_object('app.config.Config')

# Allowed file extensions for audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "App is up and running!"

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if all files are in the request
    if 'brands_file' not in request.files or 'reference_audio' not in request.files or 'input_audio' not in request.files:
        return jsonify({"error": "Missing one or more required files: 'brands_file', 'reference_audio', 'input_audio'"}), 400
    
    # Retrieve the files
    brands_file = request.files['brands_file']
    reference_audio = request.files['reference_audio']
    input_audio = request.files['input_audio']
    
    # Check file validity
    if not all(allowed_file(f.filename) for f in [brands_file, reference_audio, input_audio]):
        return jsonify({"error": "One or more files have an unsupported file type"}), 400
    
    # Secure filenames
    brands_filename = secure_filename(brands_file.filename)
    reference_audio_filename = secure_filename(reference_audio.filename)
    input_audio_filename = secure_filename(input_audio.filename)
    
    # Define paths to save the uploaded files
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    brands_path = os.path.join(upload_folder, brands_filename)
    reference_audio_path = os.path.join(upload_folder, reference_audio_filename)
    input_audio_path = os.path.join(upload_folder, input_audio_filename)
    
    # Save files
    brands_file.save(brands_path)
    reference_audio.save(reference_audio_path)
    input_audio.save(input_audio_path)
    
    # Process the input audio using the provided paths
    try:
        transcription = process_audio(
            audio_path=input_audio_path,
            output_dir=app.config['OUTPUT_FOLDER'],
            brands_file=brands_path,
            reference_audio_path=reference_audio_path,
            groq_key=app.config['GROQ_KEY'],
            hg_key=app.config['HG_KEY']
        )
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
