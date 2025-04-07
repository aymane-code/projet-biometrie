##----Description du Projet
Ce projet propose un système de reconnaissance biométrique multimodale combinant la reconnaissance faciale et vocale pour une identification plus robuste et sécurisée. Il utilise des modèles de deep learning (MTCNN, Facenet, Wav2Vec2) pour extraire des caractéristiques uniques à partir des visages et des voix, puis les compare à une base de données pré-enregistrée pour effectuer l'identification.

##---Fonctionnalités
Reconnaissance faciale en temps réel via la webcam.

Reconnaissance vocale à partir d'un enregistrement audio de 3 secondes.

Mode multimodal combinant visage et voix pour une identification plus fiable.

Interface intuitive avec Streamlit.

🛠️--------- Prérequis
Python 3.8 ou supérieur

------------## Structure du Projet

projet-biometrie/
├── app5.py                 # Script principal de l'application
├── Facemodel/              # Dossier contenant les embeddings faciaux
│   └── face_embeddings.npz
├── Audiomodel/             # Dossier contenant les embeddings audio
│   └── audio_embeddings.npz
├── requirements.txt        # Liste des dépendances
└── README.md               # Documentation du projet


--------------##Fichier requirements.txt
streamlit==1.32.0
mtcnn==0.1.1
facenet-pytorch==2.5.3
opencv-python==4.8.0.76
numpy==1.24.0
torch==2.0.0
scikit-learn==1.3.0
sounddevice==0.4.6
librosa==0.10.0
transformers==4.33.0
scipy==1.10.0

----------##Fonctionnement
Reconnaissance Faciale :

Cliquez sur "Démarrer Reconnaissance Faciale" pour lancer la détection en temps réel via la webcam.

Cliquez sur "Arrêter" pour interrompre le processus.

Reconnaissance Vocale :

Cliquez sur "Reconnaissance Vocale" pour enregistrer un échantillon vocal (3 secondes).

Le système compare l'audio aux embeddings pré-enregistrés.

##----Mode Multimodal :

Cliquez sur "Reconnaissance Multimodale" pour capturer simultanément un visage et un échantillon vocal.

Le système fusionne les résultats pour une identification plus précise.

📊 Résultats Attendus
Identification des personnes enregistrées dans la base de données.

Affichage des résultats avec un niveau de confiance.

Gestion des conflits (ex: visage et voix ne correspondent pas).

⚠️ Limitations
Performances dépendantes de la qualité de la webcam et du microphone.

Nécessite une base de données préalablement constituée (embeddings).
