import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage import color, transform, feature

# Charger le modèle
loaded_model = joblib.load('face_detection_model.sav')

# Titre et description de l'application
st.title("Détection de Visages")
st.markdown("Téléchargez une image et cliquez sur le bouton pour détecter les visages.")

# Sélectionner une image depuis l'ordinateur
uploaded_image = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

def sliding_window(img, patch_size=(62, 47), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

def draw_rectangles(image, indices):
    Ni, Nj = (62, 47)
    img_copy = np.copy(image)

    if len(indices) > 0:
        # Trouver les coordonnées du coin supérieur gauche du rectangle englobant
        top_left = (min(indices[:, 1]), min(indices[:, 0]))

        # Trouver les coordonnées du coin inférieur droit du rectangle englobant
        bottom_right = (max(indices[:, 1]) + Ni, max(indices[:, 0]) + Nj)

        # Dessiner le rectangle englobant
        img_copy[top_left[1]:top_left[1]+2, top_left[0]:bottom_right[0]] = 0  # Ligne supérieure (blanc)
        img_copy[bottom_right[1]-2:bottom_right[1], top_left[0]:bottom_right[0]] = 0  # Ligne inférieure (blanc)
        img_copy[top_left[1]:bottom_right[1], top_left[0]:top_left[0]+2] = 0  # Ligne gauche (blanc)
        img_copy[top_left[1]:bottom_right[1], bottom_right[0]-2:bottom_right[0]] = 0  # Ligne droite (blanc)

    return img_copy




if uploaded_image is not None:
    # Charger l'image et la prétraiter
    img = Image.open(uploaded_image)
    img = np.array(img)
    gray_img = color.rgb2gray(img)
    resized_img = transform.rescale(gray_img, 0.5)
    cropped_img = resized_img

    # Afficher l'image
    st.image(cropped_img, caption="Image Téléchargée", use_column_width=True)

    # Bouton pour détecter les visages
    if st.button("Détecter les visages"):
        # Extraire les patches
        indices, patches = zip(*sliding_window(cropped_img))
        patches_hog = np.array([feature.hog(patch) for patch in patches])

        # Prédire les visages
        labels = loaded_model.predict(patches_hog)

        # Dessiner les rectangles autour des visages détectés
        indices = np.array(indices)
        labeled_indices = indices[labels == 1]
        img_with_rectangles = draw_rectangles(cropped_img, labeled_indices)
        st.image(img_with_rectangles, caption="Image avec rectangles", use_column_width=True)