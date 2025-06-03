import streamlit as st
import requests
from PIL import Image
import base64
import os

st.set_page_config(
    page_title="Détection de Pneumonie par IA",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.90);
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

def set_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            color: #333336;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        h1, h2, h3 {
            color: #003366;
        }

        .stButton > button {
            background-color: #00ACC1;
            color: white;
            border-radius: 0.5rem;
            padding: 0.6rem 1.2rem;
            border: none;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #007c91;
        }

        .stMarkdown p {
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

background_path = os.path.join(os.path.dirname(__file__), "1234.JPG")
set_background(background_path)
set_custom_styles()

st.title("🩺  Détection de Pneumonie par IA")
st.write("Téléchargez une image de radiographie pulmonaire (JPG, JPEG, PNG) pour prédire si la pneumonie est présente ou non.")
st.write("---")

FLASK_API_URL = "http://127.0.0.1:5000/predict"

uploaded_file = st.file_uploader(
    "Chargez une image de radiographie pulmonaire",
    type=["jpg", "jpeg", "png"],
    help="Seules les images JPG, JPEG, PNG sont acceptées."
)


if uploaded_file is not None:
    st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)

    with st.spinner("Traitement en cours. Veuillez patienter."):
        image_bytes = uploaded_file.getvalue()
        files = {'image': (uploaded_file.name, image_bytes, uploaded_file.type)}

        try:
            response = requests.post(FLASK_API_URL, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()
                predicted_class = result.get('predicted_class')
                confidence = result.get('confidence')

                st.write("---")
                st.subheader("Résultats de la prédiction :")

                if predicted_class == "PNEUMONIA":
                    st.error("**Diagnostic : Pneumonie**")
                    st.write(f"Confiance du détection est **{confidence:.2f}%**")
                    st.warning("Cette détection est basée sur un modèle d'IA. Consultez toujours un médecin pour un diagnostic confirmé.")
                elif predicted_class == "NORMAL":
                    st.success("**Diagnostic : Normal**")
                    st.write(f"Confiance du détection est **{confidence:.2f}%**")
                    st.info("Cette détection est basée sur un modèle d'IA. Consultez toujours un médecin pour un diagnostic confirmé.")
                else:
                    st.warning("La prédiction n'a pas pu être déterminée.")

            elif response.status_code == 400:
                st.error(f"Erreur (400) : {response.json().get('error', 'Requête invalide.')}")
            elif response.status_code == 500:
                st.error(f"Erreur serveur (500) : {response.json().get('error', 'Erreur interne.')}")
            else:
                st.error(f"Erreur inattendue : Code {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Connexion à l'API Flask impossible.")
            st.warning("Assurez-vous que l'API tourne sur http://127.0.0.1:5000")
        except requests.exceptions.Timeout:
            st.error("La requête a expiré.")
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {e}")
