import os
import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch

# Initialiser les modèles
detector = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Dossier du dataset
dataset_dir = 'dataset'
embeddings = []
labels = []

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Détection du visage avec MTCNN
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            continue  # Passer si aucun visage détecté
        
        # Prendre la première face détectée
        x, y, w, h = faces[0]['box']
        face = img[y:y+h, x:x+w]
        
        # Prétraitement pour Facenet
        face_resized = cv2.resize(face, (160, 160))
        face_resized = (face_resized.astype('float32') - 127.5) / 127.5  # Normalisation
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float()
        
        # Extraire l'embedding
        with torch.no_grad():
            embedding = resnet(face_tensor).numpy()
        
        embeddings.append(embedding.flatten())
        labels.append(person_name)

# Sauvegarder les embeddings et labels
np.savez('Facemodel/face_embeddings.npz', 
         embeddings=np.array(embeddings), 
         labels=np.array(labels))

print("Embeddings sauvegardés avec succès !")