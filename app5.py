import os
# Désactiver les optimisations ONEDNN pour éviter des conflits potentiels
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st  # Framework pour créer des applications web interactives
from mtcnn import MTCNN  # Détecteur de visages basé sur les réseaux de neurones
from facenet_pytorch import InceptionResnetV1  # Modèle de reconnaissance faciale
import cv2  # Bibliothèque de vision par ordinateur
import numpy as np  # Bibliothèque pour les calculs numériques
import torch  # Framework de deep learning
from sklearn.metrics.pairwise import euclidean_distances  # Calcul des distances euclidiennes
from sklearn.preprocessing import LabelEncoder  # Encodage des labels
import sounddevice as sd  # Enregistrement audio
from scipy.io.wavfile import write  # Sauvegarde des fichiers audio
import librosa  # Traitement audio
import tempfile  # Création de fichiers temporaires
from transformers import Wav2Vec2Processor, Wav2Vec2Model  # Modèle de traitement audio

# Charger les modèles une seule fois pour optimiser les performances
@st.cache_resource
def load_models():
    # Initialiser le détecteur de visages MTCNN
    detector = MTCNN()
    
    # Charger le modèle de reconnaissance faciale Facenet pré-entraîné
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Charger les embeddings faciaux et audio pré-calculés
    face_data = np.load('Facemodel/face_embeddings.npz')
    audio_data = np.load('Audiomodel/audio_embeddings.npz')
    
    # Initialiser le modèle audio Wav2Vec2
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Retourner un dictionnaire contenant tous les modèles et données
    return {
        'detector': detector,
        'facenet': facenet_model,
        'face_embeddings': face_data['embeddings'],
        'face_labels': face_data['labels'],
        'audio_embeddings': audio_data['embeddings'],
        'audio_labels': audio_data['labels'],
        'processor': processor,
        'wav2vec': wav2vec_model
    }

# Charger les modèles et les données
models = load_models()

# Encoder les labels pour les visages et les audios
label_encoder = LabelEncoder()
face_encoded = label_encoder.fit_transform(models['face_labels'])
audio_encoded = label_encoder.transform(models['audio_labels'])

# Configuration audio
FS = 16000  # Fréquence d'échantillonnage
DURATION = 3  # Durée de l'enregistrement en secondes

# Fonction pour extraire les embeddings audio
def get_audio_embedding(audio_path):
    # Charger l'audio et le rééchantillonner
    audio, rate = librosa.load(audio_path, sr=FS)
    
    # Prétraiter l'audio avec Wav2Vec2
    inputs = models['processor'](audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    # Extraire les embeddings audio
    with torch.no_grad():
        outputs = models['wav2vec'](**inputs)
    
    # Retourner la moyenne des embeddings
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Fonction pour reconnaître un visage
def recognize_face(face_img):
    # Redimensionner et normaliser l'image du visage
    face_resized = cv2.resize(face_img, (160, 160))
    face_resized = face_resized.astype('float32')
    face_resized = (face_resized - 127.5) / 127.5
    face_resized = np.expand_dims(face_resized, axis=0)
    
    # Convertir en tenseur PyTorch
    face_tensor = torch.from_numpy(face_resized).permute(0, 3, 1, 2).float()
    
    # Extraire l'embedding du visage
    with torch.no_grad():
        embedding = models['facenet'](face_tensor).numpy()
    
    # Calculer les distances euclidiennes par rapport aux embeddings connus
    distances = euclidean_distances(embedding, models['face_embeddings'])
    min_dist_idx = np.argmin(distances)
    min_dist = distances[0, min_dist_idx]
    
    # Définir un seuil de confiance
    threshold = 1.0
    if min_dist < threshold:
        return {
            'label': label_encoder.inverse_transform([face_encoded[min_dist_idx]])[0],
            'idx': min_dist_idx
        }
    else:
        return {'label': "Unknown", 'idx': -1}

# Fonction pour la reconnaissance faciale en temps réel
def real_time_recognition():
    stframe = st.empty()  # Espace pour afficher la vidéo
    cap = cv2.VideoCapture(0)  # Capturer la vidéo depuis la webcam
    
    while st.session_state.get('running', False):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir l'image en RGB pour la détection des visages
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = models['detector'].detect_faces(rgb_frame)
        
        # Pour chaque visage détecté
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(x, 0), max(y, 0)
            
            # Extraire l'image du visage
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
                
            # Reconnaître le visage
            result = recognize_face(face_img)
            
            # Dessiner un rectangle autour du visage et afficher le label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, result['label'], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # Afficher l'image dans Streamlit
        stframe.image(frame, channels="BGR")
    
    cap.release()

# Fonction pour la reconnaissance multimodale (visage + voix)
def multimodal_recognition():
    # Capturer une image depuis la webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Erreur de capture vidéo")
        return
    
    # Détecter et extraire le visage
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = models['detector'].detect_faces(rgb_frame)
    
    if not faces:
        st.error("Aucun visage détecté")
        return
    
    face = faces[0]
    x, y, w, h = face['box']
    x, y = max(x, 0), max(y, 0)
    face_img = frame[y:y+h, x:x+w]
    
    if face_img.size == 0:
        st.error("Visage non valide")
        return
    
    # Afficher l'image capturée
    st.image(face_img, caption='Visage capturé', channels="BGR")
    
    # Reconnaître le visage
    face_result = recognize_face(face_img)
    
    # Enregistrer l'audio
    try:
        with st.spinner('Enregistrement audio...'):
            recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            
            # Sauvegarder l'audio dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_filename = tmp.name
                write(tmp_filename, FS, recording)
            
            # Extraire l'embedding audio
            audio_embedding = get_audio_embedding(tmp_filename)
            
            # Supprimer le fichier temporaire
            try:
                os.unlink(tmp_filename)
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier temporaire : {e}")
    except Exception as e:
        st.error(f"Erreur audio: {str(e)}")
        return
    
    # Combiner les résultats faciaux et audio
    combined_result = combine_modalities(face_result, audio_embedding)
    st.success(combined_result)

# Fonction pour la reconnaissance vocale seule
def audio_recognition():
    try:
        with st.spinner('Enregistrement audio...'):
            # Enregistrer l'audio
            recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            
            # Sauvegarder l'audio dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_filename = tmp.name
                write(tmp_filename, FS, recording)
            
            # Extraire l'embedding audio
            audio_embedding = get_audio_embedding(tmp_filename)
            
            # Supprimer le fichier temporaire
            try:
                os.unlink(tmp_filename)
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier temporaire : {e}")
    except Exception as e:
        st.error(f"Erreur audio: {str(e)}")
        return
    
    # Calculer les distances euclidiennes par rapport aux embeddings audio connus
    audio_distances = euclidean_distances(audio_embedding, models['audio_embeddings'])
    audio_min_idx = np.argmin(audio_distances)
    min_distance = audio_distances[0, audio_min_idx]
    
    # Identifier le label correspondant
    audio_label = label_encoder.inverse_transform([audio_encoded[audio_min_idx]])[0]
    
    # Afficher le résultat
    st.success(f"Identifié par voix: {audio_label} (Confiance: {1 - min_distance:.2f})")

# Fonction pour combiner les résultats faciaux et audio
def combine_modalities(face_result, audio_embedding):
    # Si le visage est inconnu
    if face_result['idx'] == -1:
        return "Unknown (Face)"
    
    # Calculer les distances euclidiennes pour l'audio
    audio_distances = euclidean_distances(audio_embedding, models['audio_embeddings'])
    audio_min_idx = np.argmin(audio_distances)
    
    # Identifier les labels
    face_label = label_encoder.inverse_transform([face_encoded[face_result['idx']]])[0]
    audio_label = label_encoder.inverse_transform([audio_encoded[audio_min_idx]])[0]
    
    # Comparer les labels
    if face_label == audio_label:
        return f"Identifié: {face_label} (Confiance: {1 - audio_distances[0, audio_min_idx]:.2f})"
    else:
        return f"Conflit: Visage={face_label}, Voix={audio_label}"

# Interface Streamlit
st.title("Reconnaissance Multimodale")

# Créer des colonnes pour les boutons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button('Démarrer Reconnaissance Faciale'):
        st.session_state.running = True
        real_time_recognition()

with col2:
    if st.button('Arrêter'):
        st.session_state.running = False
        st.experimental_rerun()

with col3:
    if st.button('Reconnaissance Multimodale'):
        multimodal_recognition()

with col4:
    if st.button('Reconnaissance Vocale'):
        audio_recognition()

# Afficher un avertissement si la reconnaissance faciale est en cours
if st.session_state.get('running', False):
    st.warning("La reconnaissance faciale est en cours...")