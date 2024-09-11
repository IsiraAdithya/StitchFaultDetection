import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

class ImageFaultDetector:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Stitch Fault Detection")
        self.root.geometry("500x700")
        self.root.configure(bg="#f0f0f0")

        self.setup_header()
        self.setup_image_display()
        self.setup_buttons()
        self.setup_result_display()
        self.setup_status_display()

    def setup_header(self):
        header = tk.Frame(self.root, bg="#3498db", padx=10, pady=10)
        header.pack(fill=tk.X)

        title = tk.Label(header, text="Stitch Fault Detection", font=("Arial", 20, "bold"), bg="#3498db", fg="white")
        title.pack()

    def setup_image_display(self):
        self.image_frame = tk.Frame(self.root, bg="#ffffff", width=300, height=300)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(0)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(expand=True, fill=tk.BOTH)

    def setup_buttons(self):
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        buttons = [
            ("Upload Image", self.upload_image),
            ("Load New Model", self.load_new_model),
            ("Load Label Classes", self.load_new_label_classes)
        ]

        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command, bg="#2ecc71", fg="white",
                            font=("Arial", 12), padx=10, pady=5)
            btn.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)

    def setup_result_display(self):
        result_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=10)

        self.result_label = tk.Label(result_frame, text="Prediction will appear here",
                                     font=("Arial", 14), bg="#f0f0f0", wraplength=400)
        self.result_label.pack()

    def setup_status_display(self):
        status_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=10)

        self.model_status = tk.Label(status_frame, text="Model: Not loaded", font=("Arial", 12), bg="#f0f0f0")
        self.model_status.pack()

        self.label_status = tk.Label(status_frame, text="Label Classes: Not loaded", font=("Arial", 12), bg="#f0f0f0")
        self.label_status.pack()

    def load_new_model(self):
        model_path = filedialog.askopenfilename(title="Select a model file", filetypes=[("H5 files", "*.h5")])
        if model_path:
            try:
                self.model = load_model(model_path)
                self.model_status.config(text=f"Model: Loaded from {model_path.split('/')[-1]}")
                self.show_message("Success", f"Model loaded from {model_path}")
            except Exception as e:
                self.show_message("Error", f"Failed to load model: {str(e)}")

    def load_new_label_classes(self):
        label_classes_path = filedialog.askopenfilename(title="Select label_classes.npy", filetypes=[("Numpy files", "*.npy")])
        if label_classes_path:
            try:
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.load(label_classes_path)
                self.label_status.config(text=f"Label Classes: Loaded from {label_classes_path.split('/')[-1]}")
                self.show_message("Success", f"Label classes loaded from {label_classes_path}")
            except Exception as e:
                self.show_message("Error", f"Failed to load label classes: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            if self.model is not None and self.label_encoder is not None:
                predicted_label = self.predict_image(file_path)
                self.result_label.config(text=f"Predicted class: {predicted_label}")
            else:
                self.result_label.config(text="Please load model and label classes before prediction.")
        except Exception as e:
            self.show_message("Error", f"Failed to process image: {str(e)}")

    def predict_image(self, image_path):
        if self.model is None or self.label_encoder is None:
            return "Error: Model or Label Classes not loaded."

        new_image = cv2.imread(image_path)
        if new_image is None:
            return "Error: Could not load image."

        new_image_resized = cv2.resize(new_image, (128, 128))
        new_image_normalized = new_image_resized / 255.0
        new_image_input = np.expand_dims(new_image_normalized, axis=0)

        prediction = self.model.predict(new_image_input)
        predicted_class = np.argmax(prediction)
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label

    def show_message(self, title, message):
        messagebox.showinfo(title, message)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageFaultDetector()
    app.run()