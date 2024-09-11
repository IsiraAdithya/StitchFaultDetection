from PIL import Image
import pytesseract

# Define the paths to the uploaded images
image_paths = [
    r'C:\Users\Isira Dabare\Downloads\Chat\5.png',  # Using raw string notation
    r'C:\Users\Isira Dabare\Downloads\Chat\4.png',
    r'C:\Users\Isira Dabare\Downloads\Chat\3.png'
]

# Extract text from the images using pytesseract
extracted_texts = []
for image_path in image_paths:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    extracted_texts.append(text)

# Print the extracted texts
for idx, text in enumerate(extracted_texts, 1):
    print(f"Extracted text from image {idx}:\n{text}\n")
