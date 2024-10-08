import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from pydub import AudioSegment
from pyannote.audio import Inference
import torchaudio
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from demucs.apply import apply_model
from demucs.pretrained import get_model
import librosa
from groq import Groq
import re
import time
from speechbrain.inference.speaker import SpeakerRecognition

def process_audio(audio_path, output_dir, brands_file, reference_audio_path, groq_key, hg_key):
    # Assurez-vous que le répertoire de sortie existe
    os.makedirs(output_dir, exist_ok=True)

    # Appliquer Demucs et DeepFilterNet, puis convertir en mono
    filtered_audio_path = apply_demucs_and_deepfilter(audio_path)
    mono_audio_path = convert_to_mono(filtered_audio_path, output_dir)

    # Créer un manifest pour la tâche de diarisation
    manifest_path, rttm_path = create_manifest(mono_audio_path, output_dir)

    # Exécuter la diarisation
    rttm_file, adjusted_segments, waveform, sample_rate, adjusted_segments2 = run_diarization(mono_audio_path, manifest_path, output_dir)
    print(adjusted_segments2)
    if len(adjusted_segments2) < 4:
        raise Exception('Le fichier audio contient du bruit et ne peut pas être traité.')
    else:
        # Traiter les segments
        speaker_segments = process_segments(adjusted_segments, waveform)

        # Charger les mots spéciaux depuis un fichier
        special_words = load_special_words(brands_file)
        print(special_words)

        # Charger le fichier audio complet
        full_audio = AudioSegment.from_wav(filtered_audio_path)

        # Traiter les embeddings des locuteurs et identifier la meilleure correspondance
        best_match, filtered_segments = process_speaker_embeddings(adjusted_segments, special_words, full_audio, reference_audio_path, groq_key)

        # Transcrire et corriger la conversation complète
        corrected_full_conversation = transcribe_and_correct_conversation(filtered_segments, full_audio, groq_key, brands_file)

        # Construire le chemin de sortie de la transcription
        transcription_output_filename = f"{os.path.basename(reference_audio_path)}_{os.path.basename(audio_path)}_transcription.txt"
        transcription_output_path = os.path.join(output_dir, transcription_output_filename)

        # Écrire la transcription dans le fichier
        corrected_full_conversation = corrected_full_conversation.replace("*", "")
        print(f"Transcription  : \n {corrected_full_conversation}")

        # Supprimer tout ce qui est entre parenthèses et les parenthèses elles-mêmes
        corrected_full_conversation = re.sub(r"\(.*?\)", "", corrected_full_conversation)

        with open(transcription_output_path, "w", encoding="utf-8") as f:
            speaker_lines = [line.split(":", 1)[1].strip() for line in corrected_full_conversation.split("\n") if re.match(r"(?i)^speaker\s", line)]
            filtered_discussion = "\n".join(speaker_lines)
            f.write(filtered_discussion)

        # Retourner la transcription corrigée
        return filtered_discussion

def apply_demucs_and_deepfilter(audio_path):
    print("Applying Demucs and DeepFilterNet...")
    model = get_model('htdemucs')
    wav, sr = torchaudio.load(audio_path)
    wav = wav.unsqueeze(0)  # Ajouter la dimension batch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = apply_model(model, wav.to(device))
    vocals = sources[0][-1]
    demucs_output_path = "separated_vocals.wav"
    torchaudio.save(demucs_output_path, vocals.cpu(), sr)
    enhanced_audio_path = apply_deepfilter(demucs_output_path)
    return enhanced_audio_path

def apply_deepfilter(audio_path):
    """Applique DeepFilterNet pour nettoyer l'audio."""
    filename = os.path.basename(audio_path)
    output_dir = "/tmp"  # Change this to a writable directory
    output_file = filename.replace(".mp3", "_DeepFilterNet3.mp3")
    print(f"Applying DeepFilterNet to {filename}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    os.system(f'deepFilter "{audio_path}" --output-dir {output_dir}')
    print(f"DeepFilterNet applied. Output saved as {output_file}")
    return os.path.join(output_dir, output_file)

def convert_to_mono(audio_path, output_dir):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:  # Vérifier si l'audio est multi-canaux
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convertir en mono
        mono_audio_path = os.path.join(output_dir, "mono_" + os.path.basename(audio_path))
        torchaudio.save(mono_audio_path, waveform, sample_rate)
        return mono_audio_path
    else:
        return audio_path  # Retourner le chemin original si déjà mono

def create_manifest(mono_audio_path, output_dir):
    rttm_path = os.path.join(output_dir, "pred_rttms", f"{os.path.basename(mono_audio_path).replace('.wav', '.rttm')}")
    meta = {
        'audio_filepath': mono_audio_path,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': 2,  # Mettre à jour cette valeur en fonction de votre attente
        'rttm_filepath': rttm_path,
        'uem_filepath': None
    }
    manifest_path = os.path.join(output_dir, 'input_manifest.json')
    os.makedirs(os.path.dirname(rttm_path), exist_ok=True)
    with open(manifest_path, 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')
    return manifest_path, rttm_path

def run_diarization(mono_audio_path, manifest_path, output_dir):
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    config_path = os.path.join(output_dir, 'diar_infer_telephonic.yaml')

    # Télécharger le fichier de configuration s'il n'existe pas
    if not os.path.exists(config_path):
        import wget
        config_path = wget.download(config_url, out=output_dir)

    # Charger la configuration
    config = OmegaConf.load(config_path)

    # Mettre à jour la configuration avec vos chemins
    config.diarizer.manifest_filepath = manifest_path  # Utiliser le manifest que nous venons de créer
    config.diarizer.out_dir = output_dir
    config.diarizer.speaker_embeddings.model_path = 'titanet_large'  # Utiliser le modèle pré-entraîné des locuteurs
    config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    config.diarizer.vad.parameters.onset = 0.5
    config.diarizer.vad.parameters.offset = 0.45
    config.diarizer.vad.parameters.pad_offset = -0.07
    config.diarizer.oracle_vad = False  # Désactiver VAD oracle
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Initialiser et exécuter le diarizer
    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize([mono_audio_path])

    # Traiter le fichier RTTM
    rttm_file = os.path.join(output_dir, 'pred_rttms', os.path.basename(mono_audio_path).replace('.wav', '.rttm'))
    waveform, sample_rate = torchaudio.load(mono_audio_path)

    # Analyser le fichier RTTM et ajuster les segments
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
                    # Ajouter à adjusted_segments2 sans division
                    adjusted_segments2.append((current_speaker, current_start_time, current_end_time, current_waveform))
                    # Ajouter à adjusted_segments après division
                    adjusted_segments.extend(split_long_segments(current_speaker, current_start_time, current_end_time, current_waveform, sample_rate))
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_waveform = segment_waveform

        if current_speaker is not None:
            # Ajouter le dernier segment à adjusted_segments2 sans division
            adjusted_segments2.append((current_speaker, current_start_time, current_end_time, current_waveform))
            # Ajouter le dernier segment à adjusted_segments après division
            adjusted_segments.extend(split_long_segments(current_speaker, current_start_time, current_end_time, current_waveform, sample_rate))

    return adjusted_segments, adjusted_segments2

def split_long_segments(speaker, start_time, end_time, waveform, sample_rate, max_duration=30):
    split_segments = []
    total_duration = end_time - start_time
    num_splits = int(total_duration // max_duration)
    overlap_duration = 1  # 1 seconde de chevauchement

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
        speaker = segment[0]
        combined_waveform = segment[3]

        if speaker in speaker_segments:
            speaker_segments[speaker] = torch.cat((speaker_segments[speaker], combined_waveform), dim=1)
        else:
            speaker_segments[speaker] = combined_waveform
    return speaker_segments

def load_special_words(brands_file):
    with open(brands_file, 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
    return ', '.join(content)

def process_speaker_embeddings(adjusted_segments, special_words, full_audio, reference_audio_path, groq_key):
    print(adjusted_segments)
    # Charger le modèle d'embedding pré-entraîné de SpeechBrain
    embedding_model_speechbrain = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

    # Charger l'audio de référence et obtenir l'embedding
    reference_audio = AudioSegment.from_file(reference_audio_path)
    reference_embedding = get_audio_embedding_speechbrain(reference_audio, embedding_model_speechbrain)

    # Initialiser les dictionnaires pour les similarités et les audios des locuteurs
    similarities = {}
    speaker_audios = {}

    # Combiner les segments pour chaque locuteur
    for segment in adjusted_segments:
        speaker = segment[0]
        start_time = segment[1]
        end_time = segment[2]

        # Extraire la portion pertinente de l'audio
        segment_audio = full_audio[start_time * 1000:end_time * 1000]  # Convertir les secondes en millisecondes

        # Ajouter ce segment à l'audio combiné du locuteur
        if speaker not in speaker_audios:
            speaker_audios[speaker] = AudioSegment.silent(duration=0)

        speaker_audios[speaker] += segment_audio

    # Calculer les embeddings et les similarités pour chaque audio combiné des locuteurs
    for speaker, audio in speaker_audios.items():
        # Sauvegarder le fichier audio combiné pour le locuteur
        filename = f"/app/{speaker}_combined.mp3"
        audio.export(filename, format="mp3")
        print(f"Fichier audio combiné pour {speaker} sauvegardé sous {filename}")

        # Charger le fichier audio sauvegardé
        speaker_audio = AudioSegment.from_file(filename)

        # Obtenir l'embedding pour l'audio du locuteur
        speaker_embedding = get_audio_embedding_speechbrain(speaker_audio, embedding_model_speechbrain)

        # Calculer la similarité cosinus entre l'embedding de référence et celui du locuteur
        similarity = 1 - cosine(reference_embedding, speaker_embedding)
        similarities[speaker] = similarity
        print(f"Similarité cosinus pour {speaker} : {similarity}")

    # Identifier le locuteur avec la plus haute similarité à la référence
    best_match = max(similarities, key=similarities.get)
    print(f"Le locuteur qui correspond à la référence est : {best_match}")

    # Filtrer les segments pour ne garder que ceux du meilleur locuteur
    filtered_segments = [segment for segment in adjusted_segments if segment[0] == best_match]

    return best_match, filtered_segments

def get_audio_embedding_speechbrain(audio_segment, embedding_model, sample_rate=16000, min_length=16000):
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2 ** 15)
    samples = torch.tensor(samples).unsqueeze(0)

    if audio_segment.frame_rate != sample_rate:
        samples = torch.tensor(librosa.resample(samples.numpy(), orig_sr=audio_segment.frame_rate, target_sr=sample_rate))
    samples = samples.mean(axis=0, keepdim=True)  # Assurer le mono

    if samples.size(1) < min_length:
        padding = min_length - samples.size(1)
        samples = torch.nn.functional.pad(samples, (0, padding))

    with torch.no_grad():
        embedding_model = embedding_model.to('cpu')
        samples = samples.to('cpu')
        embedding = embedding_model.encode_batch(samples)

    return embedding.squeeze()

def transcribe_and_correct_conversation(filtered_segments, full_audio, groq_key, brands_file):
    full_conversation_text = ""
    special_words = load_special_words(brands_file)
    for segment in filtered_segments:
        transcription = transcribe_segment(segment[1], segment[2], segment[0], groq_key, full_audio)
        if transcription and transcription.strip():
            full_conversation_text += f"Speaker {segment[0]}: {transcription}\n"
        else:
            print(f"Transcription pour le locuteur {segment[0]} est vide. Ignoré.")
    corrected_full_conversation = correct_transcription(full_conversation_text, groq_key, special_words)
    return corrected_full_conversation

def transcribe_segment(start_time, end_time, speaker, groq_key, full_audio):
    try:
        segment_audio = full_audio[start_time * 1000:end_time * 1000]
        temp_filename = f"temp_{speaker}.wav"
        segment_audio.export(temp_filename, format="wav")

        client = Groq(api_key=groq_key)
        with open(temp_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                prompt="Specify spelling",
                response_format="text",
                language="es",
                temperature=0.0
            )
        print(f"Transcription pour {speaker} (début : {start_time}, fin : {end_time}) : {transcription}")
        os.remove(temp_filename)
        time.sleep(5)
        return transcription.strip()

    except Exception as e:
        print(f"Erreur lors de la transcription du segment pour {speaker} (début : {start_time}, fin : {end_time}) : {str(e)}")
        return None

def correct_transcription(full_conversation_text, groq_key, special_words):
    client = Groq(api_key=groq_key)
    prompt = f"""
    Please improve this discussion by fixing any mistakes in punctuation, grammar, and formatting. Don't translate it; it's a discussion detected by a speech-to-text model, so correct any mistakes. and don't mix anything spoken, keep the order the way it is, this is important. Consider deleting the nonsense one-word spoken at one time.and dont translate. Consider these special words: {special_words}.
    give the comments between of your editing between  ( ) .
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
