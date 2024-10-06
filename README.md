# audio-processing-flask-app
 

# Audio Processing Flask App

Cette application Flask permet de traiter des fichiers audio en utilisant diverses technologies de traitement audio et de transcription.

## Installation

### Prérequis

- Docker installé sur votre machine.
- Clé API Groq (`groq_key`).
- Clé API Hugging Face (`hg_key`).

### Configuration

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/audio-processing-flask-app.git
    cd audio-processing-flask-app
    ```

2. Créez un fichier `.env` à la racine du projet et ajoutez vos clés API :
    ```env
    GROQ_KEY=your_groq_key
    HG_KEY=your_hg_key
    ```

3. Construisez l'image Docker :
    ```bash
    docker build -t audio-processing-flask-app .
    ```

4. Lancez le conteneur Docker :
    ```bash
    docker run -d -p 5000:5000 --env-file .env audio-processing-flask-app
    ```

5. Accédez à l'application via [http://localhost:5000](http://localhost:5000).

## Utilisation

1. Téléchargez un fichier audio via l'interface web.
2. L'application traitera le fichier et renverra la transcription corrigée.

## Technologies Utilisées

- Flask
- PyTorch
- NeMo
- SpeechBrain
- Groq API
- Docker

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request.

## Licence

MIT
