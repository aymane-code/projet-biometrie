import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model

app = Flask(__name__)
CORS(app)

# Initialiser les modèles
detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Charger les embeddings et labels visage
face_data = np.load('Facemodel/face_embeddings.npz')
face_embeddings = face_data['embeddings']
face_labels = face_data['labels']

# Charger les embeddings et labels audio
audio_data = np.load('Audiomodel/audio_embeddings.npz')
audio_embeddings = audio_data['embeddings']
audio_labels = audio_data['labels']

# Encoder les labels (utilisation d'un seul encoder pour les deux modalités)
label_encoder = LabelEncoder()
face_encoded = label_encoder.fit_transform(face_labels)
audio_encoded = label_encoder.transform(audio_labels)  # Utiliser transform() au lieu de fit_transform()

# Configuration audio
FS = 16000
DURATION = 3
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def get_audio_embedding(audio_path):
    audio, rate = librosa.load(audio_path, sr=FS)
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def recognize_face(face_img):
    """Reconnaît un visage à partir d'une image de visage découpée"""
    face_resized = cv2.resize(face_img, (160, 160))
    face_resized = face_resized.astype('float32')
    face_resized = (face_resized - 127.5) / 127.5  # Normalisation
    face_resized = np.expand_dims(face_resized, axis=0)
    face_tensor = torch.from_numpy(face_resized).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        embedding = facenet_model(face_tensor).numpy()
    
    # Correction: Utiliser face_embeddings au lieu de embeddings
    distances = euclidean_distances(embedding, face_embeddings)
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
    """Capture vidéo en temps réel et affiche les résultats"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Conversion de couleur corrigée
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
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
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

@app.route('/recognize', methods=['POST'])
def multimodal_recognition():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()  # Important: libérer la caméra après utilisation
    
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 400
    
    face_result = recognize_face(frame)
    
    print("Enregistrement audio...")
    try:
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write(tmp.name, FS, recording)
            audio_embedding = get_audio_embedding(tmp.name)
            os.unlink(tmp.name)  # Suppression sécurisée du fichier temporaire
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    combined_result = combine_modalities(face_result, audio_embedding)
    return jsonify({"result": combined_result})

def combine_modalities(face_result, audio_embedding):
    if face_result['idx'] == -1:
        return "Unknown (Face)"
    
    audio_distances = euclidean_distances(audio_embedding, audio_embeddings)
    audio_min_idx = np.argmin(audio_distances)
    
    # Vérification de cohérence des labels
    face_label = label_encoder.inverse_transform([face_encoded[face_result['idx']]])[0]
    audio_label = label_encoder.inverse_transform([audio_encoded[audio_min_idx]])[0]
    
    if face_label == audio_label:
        return f"Identified: {face_label} (Confiance: {1 - audio_distances[0, audio_min_idx]:.2f})"
    else:
        return f"Conflict: Face={face_label}, Voice={audio_label}"

@app.route('/gui')
def gui():
    return render_template('interface.html')

if __name__ == '__main__':
    threading.Thread(target=real_time_recognition, daemon=True).start()
    app.run(host='0.0.0.0', port=5003, debug=False)