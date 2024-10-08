# app.py

import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from pydub import AudioSegment
import torchaudio
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from demucs.apply import apply_model
from demucs.pretrained import get_model
import librosa
from speechbrain.pretrained import SpeakerRecognition
from groq import Groq  # Import for Groq transcription
import re
import time
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly include the API keys
groq_key = "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"
hg_key = "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"

# Allowed file extensions for audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models at startup
def initialize_models():
    global demucs_model
    global embedding_model_speechbrain
    global groq_client
    global diarizer_config

    # Initialize Demucs model
    demucs_model = get_model('htdemucs')
    demucs_model.eval()
    logger.info("Demucs model loaded.")

    # Initialize SpeechBrain embedding model
    embedding_model_speechbrain = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"
    )
    logger.info("SpeechBrain embedding model loaded.")

    # Initialize ClusteringDiarizer configuration
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    config_path = os.path.join('config', 'diar_infer_telephonic.yaml')
    os.makedirs('config', exist_ok=True)

    # Download the config file if it doesn't exist
    if not os.path.exists(config_path):
        import wget
        config_path = wget.download(config_url, out=config_path)
        logger.info("Diarization config downloaded.")

    # Load configuration
    diarizer_config = OmegaConf.load(config_path)
    logger.info("Diarizer configuration loaded.")

    # Initialize Groq client
    groq_client = Groq(api_key=groq_key)
    logger.info("Groq client initialized.")

# Initialize models
initialize_models()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "App is up and running!"

@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint():
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
    upload_folder = 'uploads'
    output_folder = 'outputs'
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    brands_path = os.path.join(upload_folder, brands_filename)
    reference_audio_path = os.path.join(upload_folder, reference_audio_filename)
    input_audio_path = os.path.join(upload_folder, input_audio_filename)

    # Save files
    brands_file.save(brands_path)
    reference_audio.save(reference_audio_path)
    input_audio.save(input_audio_path)

    # Define path for metrics CSV
    metrics_csv_path = os.path.join(output_folder, 'metrics.csv')

    # Process the input audio using the provided paths
    try:
        transcription = main(
            audio_path=input_audio_path,
            output_dir=output_folder,
            brands_file=brands_path,
            reference_audio_path=reference_audio_path,
            metrics_csv_path=metrics_csv_path
        )
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main(audio_path, output_dir, brands_file, reference_audio_path, metrics_csv_path):
    metrics = {}
    total_start_time = time.time()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Apply Demucs and DeepFilterNet, then convert to mono
    step_start_time = time.time()
    filtered_audio_path = apply_demucs_and_deepfilter(audio_path)
    metrics['apply_demucs_and_deepfilter'] = time.time() - step_start_time

    step_start_time = time.time()
    mono_audio_path = convert_to_mono(filtered_audio_path, output_dir)
    metrics['convert_to_mono'] = time.time() - step_start_time

    # Step 2: Create a manifest for the diarization task
    step_start_time = time.time()
    manifest_path, rttm_path = create_manifest(mono_audio_path, output_dir)
    metrics['create_manifest'] = time.time() - step_start_time

    # Step 3: Run the diarization
    step_start_time = time.time()
    rttm_file, adjusted_segments, waveform, sample_rate, adjusted_segments2 = run_diarization(mono_audio_path, manifest_path, output_dir)
    metrics['run_diarization'] = time.time() - step_start_time

    if len(adjusted_segments2) < 4:
        raise Exception('The audio has noise and cannot be processed')
    else:
        # Step 4: Process the segments
        step_start_time = time.time()
        speaker_segments = process_segments(adjusted_segments, waveform)
        metrics['process_segments'] = time.time() - step_start_time

        # Step 5: Load special words from a file
        step_start_time = time.time()
        special_words = load_special_words(brands_file)
        metrics['load_special_words'] = time.time() - step_start_time

        logger.info(f"Special words: {special_words}")

        # Load the full audio file
        full_audio = AudioSegment.from_wav(filtered_audio_path)

        # Step 6: Process speaker embeddings and identify the best match
        step_start_time = time.time()
        best_match, filtered_segments = process_speaker_embeddings(adjusted_segments, special_words, full_audio, reference_audio_path)
        metrics['process_speaker_embeddings'] = time.time() - step_start_time

        # Step 7: Transcribe and correct the full conversation
        step_start_time = time.time()
        corrected_full_conversation = transcribe_and_correct_conversation(filtered_segments, full_audio, special_words)
        metrics['transcribe_and_correct_conversation'] = time.time() - step_start_time

        # Construct the transcription output path
        transcription_output_filename = f"{os.path.basename(reference_audio_path)}_{os.path.basename(audio_path)}_transcription.txt"
        transcription_output_path = os.path.join(output_dir, transcription_output_filename)

        # Write the transcription to the file
        # Assuming 'corrected_full_conversation' contains the text with '*'
        corrected_full_conversation = corrected_full_conversation.replace("*", "")
        logger.info(f"Transcription  : \n {corrected_full_conversation}")

        # Remove anything between parentheses and the parentheses themselves
        corrected_full_conversation = re.sub(r"\(.*?\)", "", corrected_full_conversation)

        with open(transcription_output_path, "w", encoding="utf-8") as f:
            speaker_lines = [line.split(":", 1)[1].strip() for line in corrected_full_conversation.split("\n") if re.match(r"(?i)^speaker\s", line)]
            filtered_discussion = "\n".join(speaker_lines)
            f.write(filtered_discussion)

        # Output the corrected conversation
        logger.info(f"Corrected Conversation:\n{filtered_discussion}")

        total_end_time = time.time()
        metrics['total_time'] = total_end_time - total_start_time

        # Save metrics to CSV
        save_metrics_to_csv(metrics, metrics_csv_path)

        # Return the corrected transcription
        return filtered_discussion

def apply_demucs_and_deepfilter(audio_path):
    logger.info("Applying Demucs and DeepFilterNet...")
    model = demucs_model  # Use preloaded model
    wav, sr = torchaudio.load(audio_path)
    wav = wav.unsqueeze(0)  # Add the batch dimension
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = apply_model(model, wav.to(device))
    vocals = sources[0][-1]
    demucs_output_path = "separated_vocals.wav"
    torchaudio.save(demucs_output_path, vocals.cpu(), sr)
    enhanced_audio_path = apply_deepfilter(demucs_output_path)
    return enhanced_audio_path

def apply_deepfilter(audio_path):
    """Applies DeepFilterNet to clean the audio."""
    filename = os.path.basename(audio_path)
    output_file = filename.replace(".wav", "_DeepFilterNet3.wav")
    output_path = os.path.join("/tmp", output_file)
    logger.info(f"Applying DeepFilterNet to {audio_path}...")
    # Run DeepFilterNet
    command = f'deepFilter "{audio_path}" --output-dir "/tmp"'
    exit_code = os.system(command)
    if exit_code != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"DeepFilterNet failed to process the file {audio_path}")
    logger.info(f"DeepFilterNet applied. Output saved as {output_path}")
    return output_path

def convert_to_mono(audio_path, output_dir):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:  # Check if the audio is multi-channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
        mono_audio_path = os.path.join(output_dir, "mono_" + os.path.basename(audio_path))
        torchaudio.save(mono_audio_path, waveform, sample_rate)
        return mono_audio_path
    else:
        return audio_path  # Return original path if already mono

def create_manifest(mono_audio_path, output_dir):
    rttm_path = os.path.join(output_dir, "pred_rttms", f"{os.path.basename(mono_audio_path).replace('.wav', '.rttm')}")
    meta = {
        'audio_filepath': mono_audio_path,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': 2,  # Update this value based on your expectation
        'rttm_filepath': rttm_path,
        'uem_filepath' : None
    }
    manifest_path = os.path.join(output_dir, 'input_manifest.json')
    os.makedirs(os.path.dirname(rttm_path), exist_ok=True)
    with open(manifest_path, 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')
    return manifest_path, rttm_path

def run_diarization(mono_audio_path, manifest_path, output_dir):
    config = diarizer_config

    # Update the configuration with your paths
    config.diarizer.manifest_filepath = manifest_path  # Use the manifest we just created
    config.diarizer.out_dir = output_dir
    config.diarizer.speaker_embeddings.model_path = 'titanet_large'  # Use pre-trained speaker model
    config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    config.diarizer.vad.parameters.onset = 0.5
    config.diarizer.vad.parameters.offset = 0.45
    config.diarizer.vad.parameters.pad_offset = -0.07
    config.diarizer.oracle_vad = False  # Disable oracle VAD
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Set num_workers to 0 to prevent multiprocessing issues
    config.diarizer.vad.parameters.num_workers = 0
    config.diarizer.speaker_embeddings.parameters.num_workers = 0

    # Initialize and run the diarizer
    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize([mono_audio_path])

    # Process the RTTM file
    rttm_file = os.path.join(output_dir, 'pred_rttms', os.path.basename(mono_audio_path).replace('.wav', '.rttm'))
    waveform, sample_rate = torchaudio.load(mono_audio_path)

    # Parse RTTM file and adjust segments
    adjusted_segments, adjusted_segments2 = adjust_segments(rttm_file, waveform, sample_rate)
    return rttm_file, adjusted_segments, waveform, sample_rate, adjusted_segments2

def adjust_segments(rttm_file, waveform, sample_rate):
    adjusted_segments = []
    adjusted_segments2 = []
    with open(rttm_file, 'r') as f:
        current_speaker = None
        current_start_time = None
        current_end_time = None
        current_waveform = None

        for line in f:
            parts = line.strip().split()
            speaker = parts[7]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            if speaker == current_speaker:
                current_end_time = end_time
                current_waveform = torch.cat((current_waveform, segment_waveform), dim=1)
            else:
                if current_speaker is not None:
                    # Append to adjusted_segments2 without splitting
                    adjusted_segments2.append((current_speaker, current_start_time, current_end_time, current_waveform))
                    # Append to adjusted_segments after splitting
                    adjusted_segments.extend(split_long_segments(current_speaker, current_start_time, current_end_time, current_waveform, sample_rate))
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_waveform = segment_waveform

        if current_speaker is not None:
            # Append the last segment to adjusted_segments2 without splitting
            adjusted_segments2.append((current_speaker, current_start_time, current_end_time, current_waveform))
            # Append the last segment to adjusted_segments after splitting
            adjusted_segments.extend(split_long_segments(current_speaker, current_start_time, current_end_time, current_waveform, sample_rate))

    return adjusted_segments, adjusted_segments2

def split_long_segments(speaker, start_time, end_time, waveform, sample_rate, max_duration=30):
    split_segments = []
    total_duration = end_time - start_time
    num_splits = int(total_duration // max_duration)
    overlap_duration = 1  # 1 second overlap

    for i in range(num_splits + 1):
        split_start_time = start_time + i * (max_duration - overlap_duration)
        split_end_time = min(split_start_time + max_duration, end_time)

        start_sample = int((split_start_time - start_time) * sample_rate)
        end_sample = int((split_end_time - start_time) * sample_rate)
        split_waveform = waveform[:, start_sample:end_sample]

        split_segments.append({
            "speaker": speaker,
            "start_time": split_start_time,
            "end_time": split_end_time,
            "waveform": split_waveform
        })

        if split_end_time >= end_time:
            break

    return split_segments

def process_segments(adjusted_segments, waveform):
    speaker_segments = {}
    for segment in adjusted_segments:
        speaker = segment["speaker"]
        combined_waveform = segment["waveform"]

        if speaker in speaker_segments:
            speaker_segments[speaker] = torch.cat((speaker_segments[speaker], combined_waveform), dim=1)
        else:
            speaker_segments[speaker] = combined_waveform
    return speaker_segments

def load_special_words(brands_file):
    with open(brands_file, 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
    return ', '.join(content)

def process_speaker_embeddings(adjusted_segments, special_words, full_audio, reference_audio_path):
    logger.info(f"Adjusted segments: {adjusted_segments}")
    # Use preloaded embedding model
    embedding_model = embedding_model_speechbrain

    # Load the reference audio and get the embedding
    reference_audio = AudioSegment.from_file(reference_audio_path)
    reference_embedding = get_audio_embedding_speechbrain(reference_audio, embedding_model)

    # Initialize a dictionary to store the cosine similarities
    similarities = {}

    # Initialize a dictionary to store the combined audio for each speaker
    speaker_audios = {}

    # Combine segments for each speaker
    for segment in adjusted_segments:
        speaker = segment['speaker']
        start_time = segment['start_time']
        end_time = segment['end_time']

        # Extract the relevant portion of the audio
        segment_audio = full_audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds

        # Add this segment to the corresponding speaker's combined audio
        if speaker not in speaker_audios:
            speaker_audios[speaker] = AudioSegment.silent(duration=0)

        speaker_audios[speaker] += segment_audio

    # Compute embeddings and similarities for each speaker's combined audio
    for speaker, audio in speaker_audios.items():
        # Save the combined audio file for the speaker
        filename = f"{speaker}_combined.wav"
        audio.export(filename, format="wav")
        logger.info(f"Combined audio file for {speaker} saved as {filename}")

        # Load the saved audio file
        speaker_audio = AudioSegment.from_file(filename)

        # Get the embedding for the speaker audio
        speaker_embedding = get_audio_embedding_speechbrain(speaker_audio, embedding_model)

        # Compute the cosine similarity between the reference and speaker embedding
        similarity = 1 - cosine(reference_embedding, speaker_embedding)
        similarities[speaker] = similarity
        logger.info(f"Cosine similarity for {speaker}: {similarity}")

    # Identify the speaker with the highest similarity to the reference
    best_match = max(similarities, key=similarities.get)
    logger.info(f"The speaker that matches the reference is: {best_match}")

    # Filter the segments to keep only those of the best-matching speaker
    filtered_segments = [segment for segment in adjusted_segments if segment['speaker'] == best_match]

    return best_match, filtered_segments

def get_audio_embedding_speechbrain(audio_segment, embedding_model, sample_rate=16000, min_length=16000):
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2 ** 15)
    samples = torch.tensor(samples).unsqueeze(0)

    if audio_segment.frame_rate != sample_rate:
        samples = torch.tensor(librosa.resample(samples.numpy(), orig_sr=audio_segment.frame_rate, target_sr=sample_rate))
    samples = samples.mean(axis=0, keepdims=True)  # Ensure mono

    if samples.size(1) < min_length:
        padding = min_length - samples.size(1)
        samples = torch.nn.functional.pad(samples, (0, padding))

    with torch.no_grad():
        embedding_model = embedding_model.to('cpu')
        samples = samples.to('cpu')
        embedding = embedding_model.encode_batch(samples)

    return embedding.squeeze().numpy()

def transcribe_and_correct_conversation(filtered_segments, full_audio, special_words):
    full_conversation_text = ""
    for segment in filtered_segments:
        transcription = transcribe_segment(segment['start_time'], segment['end_time'], segment['speaker'], full_audio)
        if transcription and transcription.strip():
            full_conversation_text += f"Speaker {segment['speaker']}: {transcription}\n"
        else:
            logger.info(f"Transcription for speaker {segment['speaker']} is empty. Skipping.")
    corrected_full_conversation = correct_transcription(full_conversation_text, special_words)
    return corrected_full_conversation

def transcribe_segment(start_time, end_time, speaker, full_audio):
    try:
        segment_audio = full_audio[start_time * 1000:end_time * 1000]
        temp_filename = f"temp_{speaker}.wav"
        segment_audio.export(temp_filename, format="wav")

        client = groq_client
        with open(temp_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                prompt="Specify spelling",
                response_format="text",
                language="es",
                temperature=0.0
            )
        logger.info(f"Transcription for {speaker} (start: {start_time}, end: {end_time}): {transcription}")
        os.remove(temp_filename)
        time.sleep(5)
        return transcription.strip()

    except Exception as e:
        logger.error(f"Error transcribing segment for {speaker} (start: {start_time}, end: {end_time}): {str(e)}")
        return None

def correct_transcription(full_conversation_text, special_words):
    client = groq_client
    prompt = f"""
    Please improve this discussion by fixing any mistakes in punctuation, grammar, and formatting. Don't translate it; it's a discussion detected by a speech-to-text model, so correct any mistakes, and don't change the order of anything spoken. Consider deleting the nonsense one-word spoken at one time, and don't translate. Consider these special words: {special_words}.
    Give the comments of your editing between ( ).
    Here is the transcription:
    {full_conversation_text}

    Please provide the most accurate and clear transcription.
    """

    corrected_discussion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="llama3-70b-8192",
    )

    output_text = corrected_discussion.choices[0].message.content.strip()

    return output_text

def save_metrics_to_csv(metrics, csv_path):
    fieldnames = ['step', 'time_elapsed']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for step, time_elapsed in metrics.items():
            writer.writerow({'step': step, 'time_elapsed': time_elapsed})

if __name__ == '__main__':
    app.run(debug=False)
