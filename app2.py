import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Initialiser les modèles
detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Charger les embeddings et labels
data = np.load('Facemodel/face_embeddings.npz')
embeddings = data['embeddings']
labels = data['labels']

# Encoder les labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

def recognize_face(face_img):
    """Reconnaît un visage à partir d'une image de visage découpée"""
    face_resized = cv2.resize(face_img, (160, 160))
    face_resized = face_resized.astype('float32')
    face_resized = (face_resized - 127.5) / 127.5  # Normalisation
    face_resized = np.expand_dims(face_resized, axis=0)
    face_tensor = torch.from_numpy(face_resized).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        embedding = facenet_model(face_tensor).numpy()
    
    distances = euclidean_distances(embedding, embeddings)
    min_dist_idx = np.argmin(distances)
    min_dist = distances[0, min_dist_idx]
    
    threshold = 1.0
    if min_dist < threshold:
        return label_encoder.inverse_transform([encoded_labels[min_dist_idx]])[0]
    else:
        return "Unknown"

def real_time_recognition():
    """Capture vidéo en temps réel et affiche les résultats"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détection des visages
        faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for face in faces:
            x, y, w, h = face['box']
            
            # Ajuster les coordonnées si nécessaires
            x, y = abs(x), abs(y)
            
            # Découper et reconnaître le visage
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
                
            label = recognize_face(face_img)
            
            # Dessiner les résultats
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return "Serveur de reconnaissance faciale actif - Caméra en cours d'exécution..."

# Lancer la reconnaissance faciale dans un thread séparé
if __name__ == '__main__':
    # Désactiver le mode debug pour éviter les problèmes de threads
    threading.Thread(target=real_time_recognition, daemon=True).start()
    app.run(host='0.0.0.0', port=5003, debug=False)