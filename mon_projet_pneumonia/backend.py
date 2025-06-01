import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Initialisation de l'application Flask ---
# Cette ligne DOIT être la première instruction liée à Flask
# avant toute définition de route (@app.route)
app = Flask(__name__)
CORS(app) # Active CORS pour permettre les requêtes depuis Streamlit

# --- Configuration du modèle ---
# Chemin vers votre modèle sauvegardé
MODEL_PATH = 'best_chest_xray_model.keras' # Assurez-vous que ce fichier est dans le même répertoire que ce script

# Dimensions d'entrée attendues par votre modèle
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Noms des classes (doivent correspondre à l'ordre d'entraînement de votre modèle)
# D'après votre notebook, 'NORMAL': 0, 'PNEUMONIA': 1
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Charger le modèle une seule fois au démarrage de l'application
try:
    model = load_model(MODEL_PATH)
    # Re-compiler le modèle est une bonne pratique après le chargement,
    # surtout si vous voulez continuer l'entraînement ou si le modèle a des couches personnalisées.
    # Pour la prédiction seule, ce n'est pas strictement nécessaire si le modèle était déjà compilé
    # avec un optimiseur et une loss compatibles.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Modèle '{MODEL_PATH}' chargé avec succès.")
except Exception as e:
    print(f"ATTENTION: Erreur lors du chargement du modèle. L'API ne pourra pas faire de prédictions: {e}")
    model = None # Le modèle n'a pas pu être chargé, l'API ne pourra pas faire de prédictions

# --- Fonction de prétraitement de l'image ---
def preprocess_image(image_bytes):
    """
    Prétraite une image pour qu'elle soit compatible avec le modèle CNN.
    Args:
        image_bytes: Les bytes de l'image (ex: depuis request.files['image'].read()).
    Returns:
        Un tableau NumPy de l'image prétraitée, prêt pour la prédiction du modèle.
    """
    try:
        # Ouvrir l'image et la convertir en RGB (certaines images peuvent être en RGBA ou niveaux de gris)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH)) # Redimensionner à la taille attendue par le modèle
        img_array = np.array(img) # Convertir en tableau NumPy
        img_array = np.expand_dims(img_array, axis=0) # Ajouter une dimension de batch (1, H, W, C)
        img_array = img_array / 255.0 # Normaliser les pixels à [0, 1]
        return img_array
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image : {e}")
        return None

# --- Route de prédiction ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Le modèle n\'a pas pu être chargé au démarrage du serveur. Veuillez vérifier le chemin du modèle et les logs du serveur.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Aucun fichier image fourni dans la requête'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné ou nom de fichier vide'}), 400

    if file:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        if processed_image is None:
            return jsonify({'error': 'Échec du prétraitement de l\'image. Le fichier est-il une image valide ?'}), 500

        try:
            # Effectuer la prédiction
            # Votre modèle est binaire (NORMAL/PNEUMONIA) avec une activation sigmoïde en sortie,
            # donc il renvoie une seule valeur entre 0 et 1.
            prediction_raw = model.predict(processed_image)[0][0]

            # Déterminer la classe prédite et la confiance
            # D'après votre notebook, 'NORMAL': 0, 'PNEUMONIA': 1
            if prediction_raw >= 0.5: # Si la probabilité pour PNEUMONIA est >= 0.5
                predicted_class = CLASS_NAMES[1] # C'est PNEUMONIA
                confidence = float(prediction_raw)
            else: # Sinon, c'est NORMAL
                predicted_class = CLASS_NAMES[0] # C'est NORMAL
                confidence = float(1 - prediction_raw) # La confiance pour NORMAL est 1 - probabilité_pneumonia

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2) # Afficher la confiance en pourcentage
            }), 200

        except Exception as e:
            # Gérer les erreurs génériques lors de la prédiction
            return jsonify({'error': f'Erreur inattendue lors de la prédiction du modèle : {str(e)}'}), 500

    # Cas où le fichier n'est pas géré ou une erreur non spécifique se produit
    return jsonify({'error': 'Erreur inconnue lors du traitement du fichier'}), 500

# --- Lancement du serveur Flask ---
if __name__ == '__main__':
    # Lance l'application Flask sur le port 5000
    # debug=True est utile en développement pour le rechargement automatique
    # et les messages d'erreur détaillés. Mettez-le à False en production.
    app.run(debug=True, port=5000)