import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Charger le modèle Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Fonction pour extraire les embeddings audio
def get_audio_embedding(audio_path):
    # Charger l'audio et le rééchantillonner à 16 kHz
    audio, rate = librosa.load(audio_path, sr=16000)
    
    # Prétraiter l'audio avec Wav2Vec2
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    # Extraire les embeddings
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    
    # Moyenner les embeddings sur la séquence temporelle
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Dossier contenant les données audio
dataset_path = "AudioDataset"

# Initialiser les listes pour les embeddings et les labels
audio_embeddings = []
audio_labels = []

# Parcourir le dossier des données audio
for label in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, label)
    
    if os.path.isdir(person_folder):
        for audio_file in os.listdir(person_folder):
            audio_path = os.path.join(person_folder, audio_file)
            
            # Extraire l'embedding audio
            embedding = get_audio_embedding(audio_path)
            
            # Ajouter l'embedding et le label aux listes
            audio_embeddings.append(embedding)
            audio_labels.append(label)

# Convertir les listes en tableaux NumPy
audio_embeddings = np.vstack(audio_embeddings)
audio_labels = np.array(audio_labels)

# Sauvegarder les embeddings et les labels dans un fichier .npz
np.savez("Audiomodel/audio_embeddings.npz", embeddings=audio_embeddings, labels=audio_labels)

print("Fichier audio_embeddings.npz créé avec succès !")