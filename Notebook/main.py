from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load the trained model and processor
model = ViTForImageClassification.from_pretrained("sagar27kumar/sagarsahu_ECG-XRAY-ViT")
processor = ViTImageProcessor.from_pretrained("sagar27kumar/sagarsahu_ECG-XRAY-ViT")

# Move model to MPS or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Function to classify an image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    label = model.config.id2label[predictions.item()]
    return label

# Test with a user-provided image
image_path = "/Users/sagarkumarsahu/Desktop/Heart-disease-prediction-using-ECG/image.jpg"  # Replace with the actual image path
predicted_label = classify_image(image_path)
print(f"Predicted Label: {predicted_label}")
