# webcam_predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.cost_estimator import estimate_cost
from utils.recommendation_engine import get_recommendation

# Load model and define class labels
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

# Preprocess frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Start webcam
cap = cv2.VideoCapture(0)

print("🎥 Press 'Space' to predict, 'Esc' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("E-Waste Webcam Prediction", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break
    elif key == 32:  # SPACE key
        input_array = preprocess_frame(frame)
        preds = model.predict(input_array)
        predicted_label = class_labels[np.argmax(preds[0])]
        cost = estimate_cost(predicted_label)
        recommendation = get_recommendation(predicted_label)

        print(f"\n🔍 Predicted Item: {predicted_label}")
        print(f"💰 Estimated Cost: ₹{cost}")
        print(f"♻️ Recommendation: {recommendation}\n")

cap.release()
cv2.destroyAllWindows()
