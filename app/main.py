from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from .helpers import process_audio

app = Flask(__name__)
app.config.from_object('app.config.Config')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(upload_path)
        
        brands_file = 'brands.txt'
        reference_audio_path = 'path/to/reference_audio.wav'  # Mettre à jour avec le chemin réel

        try:
            transcription = process_audio(
                audio_path=upload_path,
                output_dir=app.config['OUTPUT_FOLDER'],
                brands_file=brands_file,
                reference_audio_path=reference_audio_path,
                groq_key=app.config['GROQ_KEY'],
                hg_key=app.config['HG_KEY']
            )
            return jsonify({"transcription": transcription}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
