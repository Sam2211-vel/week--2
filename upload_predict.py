# upload_predict.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from utils.cost_estimator import estimate_cost
from utils.recommendation_engine import get_recommendation

# Load model and class labels
model = load_model("models/efficientnetv2b0_e_waste.h5", compile=False)

class_labels = [
    'Battery',
    'Keyboard',
    'Microwave',
    'Mobile Phone',
    'Mouse',
    'PCB',
    'Player',
    'Printer',
    'Television',
    'Washing Machine'
]

# Predict function
def predict_from_file(img_path):
    if not os.path.exists(img_path):
        print("❌ File not found.")
        return

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_label = class_labels[np.argmax(preds[0])]
    cost = estimate_cost(predicted_label)
    recommendation = get_recommendation(predicted_label)

    print(f"\n📷 File: {img_path}")
    print(f"🔍 Predicted Item: {predicted_label}")
    print(f"💰 Estimated Cost: ₹{cost}")
    print(f"♻️ Recommendation: {recommendation}")

# Example usage
if __name__ == "__main__":
    img_path = input("🖼️ Enter image path: ")
    predict_from_file(img_path)
