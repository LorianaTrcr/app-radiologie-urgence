import os
import sys
import subprocess

# Hack infaillible pour Streamlit Cloud : forcer l'environnement à utiliser la version "headless"
try:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
    subprocess.call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
except Exception as e:
    pass

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Configuration de la page ──
st.set_page_config(
    page_title="Détecteur de Fractures Osseuses",
    page_icon="🔬",
    layout="wide"
)

# ── Constantes Métier ──
CLASS_NAMES  = ['fractured', 'not fractured']
FRACTURE_IDX = 0  # Index de la classe positive 'fractured'
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Transformation de validation requise par le modèle
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Chargement optimisé du modèle depuis le dossier local ──
@st.cache_resource
def load_model():
    # On charge le modèle depuis le dossier local "./model_weights"
    model = AutoModelForImageClassification.from_pretrained(
        "./model_weights", 
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    
    # CRUCIAL : Activer les gradients pour que GradCAM fonctionne
    for param in model.parameters():
        param.requires_grad = True
        
    model.eval()
    return model

# ── Wrapper obligatoire pour l'architecture Hugging Face ──
class ResNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ── Fonction d'explicabilité GradCAM ──
def generate_gradcam(model, img_tensor):
    wrapper = ResNetWrapper(model)
    target_layer = [model.resnet.encoder.stages[-1]]
    
    with GradCAM(model=wrapper, target_layers=target_layer) as cam:
        targets = [ClassifierOutputTarget(FRACTURE_IDX)]
        grayscale = cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)
        
    return grayscale[0]

# ── Sidebar (Informations contextuelles et Disclaimer) ──
st.sidebar.title("🔬 Paramètres & Contexte")

# 1. Mini présentation / intro (Texte classique)
st.sidebar.markdown(
    "Application web de démonstration médicale. Cet outil extrait les caractéristiques "
    "visuelles via GradCAM pour expliciter la prise de décision de l'IA."
)

st.sidebar.markdown("---")

# 2. Disclaimer bien visible dans l'encadré rouge
st.sidebar.error(
    "⚕️ **DISCLAIMER CLINIQUE**\n\n"
    "Cette application est un prototype de recherche académique. "
    "Elle ne remplace en aucun cas l'analyse médicale expertisée d'un médecin radiologue. "
    "Tout diagnostic doit être rigoureusement validé par un praticien."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Architecture :** ResNet-50 Fine-tuné")
st.sidebar.markdown("**Hébergement Modèle :** GitHub")
st.sidebar.markdown("**Classes cibles :** `fractured` | `not fractured`")
st.sidebar.markdown("**Seuil d'alerte fracture :** `30%` *(Haute sensibilité)*") # Mise en avant de l'amélioration !

# ── Page Principale ──
st.title("🦴 Radiologie d'Urgence — Détection de Fractures")
st.markdown("Système d'aide au diagnostic. Téléversez une radiographie au format standard pour analyse.")

# Widget d'importation
uploaded_file = st.file_uploader(
    "Sélectionner une radiographie numérique",
    type=["jpg", "jpeg", "png"],
    help="Sélectionnez un fichier JPG, JPEG ou PNG propre."
)

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file).convert("RGB")

    # Organisation de l'espace de travail en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Cliché Radiographique Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 Analyse Automatique & Métriques")

        with st.spinner("Analyse et calcul de GradCAM en cours..."):
            # 1. Charger le modèle
            model = load_model()
            
            # 2. Appliquer les transformations
            img_tensor = VAL_TRANSFORM(image)

            # 3. Prédire
            with torch.no_grad():
                outputs = model(pixel_values=img_tensor.unsqueeze(0))
                probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()

            frac_prob = float(probs[FRACTURE_IDX])
            
            # --- EXPÉRIMENTATION (OPTION D) : SEUIL MODIFIÉ ---
            SEUIL_URGENCE = 0.50
            
            if frac_prob >= SEUIL_URGENCE:
                pred_idx = FRACTURE_IDX
                confidence = frac_prob
            else:
                pred_idx = 1 # Index pour 'not fractured'
                confidence = float(probs[1])
                
            pred_class = CLASS_NAMES[pred_idx]
            # --------------------------------------------------

            # 4. Générer GradCAM
            img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
            heatmap = generate_gradcam(model, img_tensor)
            overlay = show_cam_on_image(img_np, heatmap, use_rgb=True)

        # 5. Affichage des résultats
        if pred_class == 'fractured':
            st.error("🚨 Alerte : Signe de fracture détecté")
            st.metric(
                label="Diagnostic Principal", 
                value="FRACTURE (Positive)", 
                delta=f"Confiance : {confidence:.2%}",
                delta_color="inverse"
            )
        else:
            st.success("✅ Cliché sain : Aucun signe évident de fracture")
            st.metric(
                label="Diagnostic Principal", 
                value="SAIN (Négatif)", 
                delta=f"Confiance : {confidence:.2%}",
                delta_color="normal"
            )

        # Barre de progression
        st.markdown(f"**Niveau de suspicion clinique (Probabilité brute) :** {frac_prob:.2%}")
        st.progress(frac_prob)

        st.markdown("---")
        
        # Affichage GradCAM
        st.markdown("**🔍 Carte d'explicabilité visuelle (GradCAM) :**")
        st.image(overlay, use_container_width=True, caption="Les foyers d'activation rouges indiquent les zones décisives pour l'algorithme.")

else:
    st.info("💡 En attente de données : Veuillez importer une radiographie.")

# --- SECTION TOUJOURS VISIBLE - BAS DE PAGE ---
st.markdown("---") # Petite ligne de séparation visuelle

# st.expander crée le menu déroulant
with st.expander("❓ Comment ça marche ? (Explication du modèle)"):
    # st.info crée l'encadré bleu à l'intérieur
    st.info("""
    **Bienvenue sur notre outil d'IA en Radiologie.**
    
    1. **L'Analyse :** Lorsque vous envoyez une image, elle est lue par notre modèle **ResNet-50**, un réseau de neurones artificiels entraîné sur des milliers de radiographies (saines et fracturées).
    2. **La Décision :** L'IA cherche des "motifs" (patterns) invisibles à l'œil nu qui correspondent à des fractures osseuses et calcule un score de confiance. Pour minimiser les risques d'erreurs médicales (faux négatifs), **le seuil de détection a été abaissé à 30%**.
    3. **L'Explicabilité (GradCAM) :** Parce qu'un médecin ne peut pas faire confiance à une "boîte noire", l'outil génère une carte de chaleur (heatmap). Les zones **rouges** vous montrent exactement où l'IA a "regardé" pour prendre sa décision !
    """)
