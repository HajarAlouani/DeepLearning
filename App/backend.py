import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app) 

MODEL_PATH = 'best_chest_xray_model.keras' 
IMG_HEIGHT, IMG_WIDTH = 128, 128

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# calling/uploading the model 
try:
    model = load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Modèle '{MODEL_PATH}' chargé avec succès.")
except Exception as e:
    print(f"ATTENTION: Erreur lors du chargement du modèle. {e}")
    model = None 


# preprocessing the image 
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH)) 
        img_array = np.array(img) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image : {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error':'No model detected'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Please upload an image'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné ou nom de fichier vide'}), 400

    if file:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        if processed_image is None:
            return jsonify({'error': 'failed to  processe the image'}), 500

        try:
            prediction_raw = model.predict(processed_image)[0][0]

            # If the prediction_raw >= 0.5, the patient has pneumonia, otherwise, the prediction indicates a normal patient.
            if prediction_raw >= 0.5:
                predicted_class = CLASS_NAMES[1]
                confidence = float(prediction_raw)
            else: 
                predicted_class = CLASS_NAMES[0]
                confidence = float(1 - prediction_raw) 

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2) 
            }), 200

        except Exception as e:
            return jsonify({'error': f'Erreur inattendue lors de la prédiction du modèle : {str(e)}'}), 500

    return jsonify({'error': 'Erreur inconnue lors du traitement du fichier'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)