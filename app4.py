import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Initialiser les modèles une seule fois
@st.cache_resource
def load_models():
    detector = MTCNN()
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Charger les embeddings
    face_data = np.load('Facemodel/face_embeddings.npz')
    audio_data = np.load('Audiomodel/audio_embeddings.npz')
    
    # Initialiser le modèle audio
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
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

models = load_models()
label_encoder = LabelEncoder()
face_encoded = label_encoder.fit_transform(models['face_labels'])
audio_encoded = label_encoder.transform(models['audio_labels'])

# Configuration audio
FS = 16000
DURATION = 3

def get_audio_embedding(audio_path):
    audio, rate = librosa.load(audio_path, sr=FS)
    inputs = models['processor'](audio, sampling_rate=rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = models['wav2vec'](**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def recognize_face(face_img):
    face_resized = cv2.resize(face_img, (160, 160))
    face_resized = face_resized.astype('float32')
    face_resized = (face_resized - 127.5) / 127.5
    face_resized = np.expand_dims(face_resized, axis=0)
    face_tensor = torch.from_numpy(face_resized).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        embedding = models['facenet'](face_tensor).numpy()
    
    distances = euclidean_distances(embedding, models['face_embeddings'])
    min_dist_idx = np.argmin(distances)
    min_dist = distances[0, min_dist_idx]
    
    threshold = 1.0
    if min_dist < threshold:
        return {
            'label': label_encoder.inverse_transform([face_encoded[min_dist_idx]])[0],
            'idx': min_dist_idx
        }
    else:
        return {'label': "Unknown", 'idx': -1}

def real_time_recognition():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    while st.session_state.get('running', False):
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = models['detector'].detect_faces(rgb_frame)
        
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(x, 0), max(y, 0)
            
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
                
            result = recognize_face(face_img)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, result['label'], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        stframe.image(frame, channels="BGR")
    
    cap.release()

def multimodal_recognition():
    # Capture visage
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Erreur de capture vidéo")
        return
    
    face_result = recognize_face(frame)
    
    # Capture audio
    try:
        with st.spinner('Enregistrement audio...'):
            recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            
            # Créer un fichier temporaire et s'assurer qu'il est correctement fermé
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_filename = tmp.name  # Sauvegarder le nom du fichier
                write(tmp_filename, FS, recording)
            
            # Charger l'audio et calculer l'embedding
            audio_embedding = get_audio_embedding(tmp_filename)
            
            # Supprimer le fichier temporaire après utilisation
            try:
                os.unlink(tmp_filename)
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier temporaire : {e}")
    except Exception as e:
        st.error(f"Erreur audio: {str(e)}")
        return
    
    # Combinaison des résultats
    combined_result = combine_modalities(face_result, audio_embedding)
    st.success(combined_result)

def combine_modalities(face_result, audio_embedding):
    if face_result['idx'] == -1:
        return "Unknown (Face)"
    
    audio_distances = euclidean_distances(audio_embedding, models['audio_embeddings'])
    audio_min_idx = np.argmin(audio_distances)
    
    face_label = label_encoder.inverse_transform([face_encoded[face_result['idx']]])[0]
    audio_label = label_encoder.inverse_transform([audio_encoded[audio_min_idx]])[0]
    
    if face_label == audio_label:
        return f"Identifié: {face_label} (Confiance: {1 - audio_distances[0, audio_min_idx]:.2f})"
    else:
        return f"Conflit: Visage={face_label}, Voix={audio_label}"

# Interface Streamlit
st.title("Reconnaissance Multimodale")

col1, col2, col3 = st.columns(3)
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

if st.session_state.get('running', False):
    st.warning("La reconnaissance faciale est en cours...")