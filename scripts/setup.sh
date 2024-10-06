#!/bin/bash

# Installer les dépendances système
apt-get update && apt-get install -y \
    sox \
    libsndfile1 \
    ffmpeg

# Installer les dépendances Python
pip install -r requirements.txt
