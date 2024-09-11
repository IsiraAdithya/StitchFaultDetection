import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Global variables to hold the loaded model and label encoder
model = None
label_encoder = None

# Function to load a new model
def load_new_model():
    global model
    model_path = filedialog.askopenfilename(title="Select a model file", filetypes=[("H5 files", "*.h5")])
    if model_path:
        model = load_model(model_path)
        result_label.config(text=f"Model loaded from {model_path}")

# Function to load a new label_classes.npy file
def load_new_label_classes():
    global label_encoder
    label_encoder = LabelEncoder()
    label_classes_path = filedialog.askopenfilename(title="Select label_classes.npy", filetypes=[("Numpy files", "*.npy")])
    if label_classes_path:
        label_encoder.classes_ = np.load(label_classes_path)
        result_label.config(text=f"Label classes loaded from {label_classes_path}")

# Function to predict the class of an uploaded image
def predict_image(image_path):
    if model is None or label_encoder is None:
        return "Error: Model or Label Classes not loaded."

    new_image = cv2.imread(image_path)
    if new_image is None:
        return "Error: Could not load image."

    # Step 1: Resize the image to match the input size of the model (128x128)
    new_image_resized = cv2.resize(new_image, (128, 128))

    # Step 2: Normalize the pixel values to [0, 1]
    new_image_normalized = new_image_resized / 255.0

    # Step 3: Add a batch dimension (1, 128, 128, 3)
    new_image_input = np.expand_dims(new_image_normalized, axis=0)

    # Step 4: Make a prediction
    prediction = model.predict(new_image_input)

    # Step 5: Get the predicted class index
    predicted_class = np.argmax(prediction)

    # Step 6: Map the predicted class index back to the class label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label

# Function to upload an image and display it
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return  # No file selected

    # Open the image and display it
    img = Image.open(file_path)
    img = img.resize((250, 250))  # Resize the image for display
    img_tk = ImageTk.PhotoImage(img)

    # Update the image display
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Make prediction and update label
    predicted_label = predict_image(file_path)
    result_label.config(text=f"Predicted class: {predicted_label}")

# Create the main window
root = tk.Tk()
root.title("Image Fault Detection")
root.geometry("400x600")

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Create a button to load a new model
load_model_button = tk.Button(root, text="Load New Model", command=load_new_model)
load_model_button.pack(pady=10)

# Create a button to load new label_classes.npy
load_label_button = tk.Button(root, text="Load Label Classes", command=load_new_label_classes)
load_label_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 14))
result_label.pack(pady=10)

# Start the GUI loop
root.mainloop()
