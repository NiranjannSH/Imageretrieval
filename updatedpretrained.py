import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from scipy.spatial.distance import cosine
import os

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Define the directory containing the images
img_dir = "C://4th_sem_project//img"

# Get the list of files in the directory
files = os.listdir(img_dir)

# Filter the list to only include image files
img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
img_files = [os.path.join(img_dir, f) for f in files if os.path.splitext(f)[1].lower() in img_extensions]

# Process the features for each image
database_features = []
features_filename = 'C://4th_sem_project//database_features.npy'

# Check if saved features exist, load them; otherwise, compute and save
if os.path.exists(features_filename):
    database_features = np.load(features_filename)
else:
    for img_file in img_files:
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array).flatten()
        database_features.append(features)
    # Save the computed features to a file
    np.save(features_filename, database_features)

# Define the Streamlit app
st.title("Image Retrieval using vgg-16 ")
st.text("Upload an image to find similar images in the dataset")

# Add image upload widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Handle image upload and retrieval
if uploaded_file is not None:
    # Load query image and extract features
    query_img = Image.open(uploaded_file).resize((224, 224))
    query_img_array = image.img_to_array(query_img)
    query_img_array = np.expand_dims(query_img_array, axis=0)
    query_img_array = preprocess_input(query_img_array)
    query_features = model.predict(query_img_array).flatten()

    # Calculate similarity scores between query image and images in the database
    similarities = []
    for db_feature in database_features:
        similarity = 1 - cosine(query_features, db_feature)
        similarities.append(similarity)

    # Retrieve the top-k images with highest similarity scores
    k = 4
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_images = [img_files[i] for i in top_k_indices]

    if len(top_k_images) > 0:
        # Display the query image
        st.subheader("Query image")
        st.image(query_img, width=300)

        # Display the top-k images
        for i, image_path in enumerate(top_k_images):
            st.subheader(f"Similar image {i+1}")
            st.image(Image.open(image_path), width=300)
    else:
        st.text("No similar images found in the dataset.")
