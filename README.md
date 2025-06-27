# week--2
# week1-ai-with-green-technology
e-waste classification
E-Waste-Classification-Project/
│
├── 📁 data/
│   ├── 📁 train/                        ← Training images organized in subfolders per class
│   ├── 📁 validation/                   ← Validation images organized similarly
│   └── 📁 test/                         ← Test images for evaluation
│
├── 📁 models/
│   └── efficientnetv2b0_e_waste.h5     ← Trained model file (saved after training)
│
├── 📁 utils/
│   ├── cost_estimator.py               ← Python module to estimate recycling cost
│   └── recommendation_engine.py        ← Python module to generate e-waste recycling suggestions
│
├── 📁 static/
│   ├── 📁 images/                       ← Stores uploaded or webcam-captured images
│   └── predictions.csv                 ← Prediction history with image name, class, cost, recommendations
│
│
├── 📄 E-Waste Generation Classification.ipynb  ← ✅ MAIN NOTEBOOK:
│   ├── 1. Project Setup and Imports
│   ├── 2. Data Loading and Preprocessing
│   ├── 3. Exploratory Data Analysis (EDA)
│   ├── 4. Model Building (EfficientNetV2B0)
│   ├── 5. Model Training & Evaluation (with accuracy/loss curves)
│   ├── 6. Confusion Matrix Visualization
│   ├── 7. Save Model
│   ├── 8. Cost Estimator & Recommendations Integration
│   ├── 9. Gradio UI (Upload + Webcam)
│   ├── 10. Prediction History Logging
│   └── 11. Final Output and Summary
│
├── 📄 webcam_predict.py                ← Standalone webcam prediction script using trained model
├── 📄 upload_predict.py                ← Standalone image upload prediction script
│
├── 📄 requirements.txt                 ← List of all required libraries (tensorflow, gradio, opencv, etc.)
├── 📄 README.md                        ← Project overview, setup steps, features, usage, results, credits
                           WEEK-1(DATASET)
                            1. Project Setup and Imports
                            code:
                            # Install necessary libraries (only run if not already installed)
!pip install tensorflow gradio opencv-python-headless matplotlib seaborn pillow scikit-learn --quiet


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image
from datetime import datetime

#(TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully!")

                                       2. Data Loading and Preprocessing

                                       # Define image size and paths
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_path = 'data/train'
val_path = 'data/validation'
test_path = 'data/test'

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_data = train_datagen.flow_from_directory(train_path, target_size=IMG_SIZE,
                                               batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_path, target_size=IMG_SIZE,
                                           batch_size=BATCH_SIZE, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_path, target_size=IMG_SIZE,
                                             batch_size=BATCH_SIZE, class_mode='categorical',
                                             shuffle=False)

class_names = list(train_data.class_indices.keys())
print("🗂️ Classes:", class_names)

                                 3. Exploratory Data Analysis (EDA)
# Count number of images per class in train/validation/test sets
import os
from collections import Counter

def count_images(directory):
    label_counts = {}
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            label_counts[label] = len(os.listdir(label_path))
    return label_counts

train_counts = count_images(train_path)
val_counts = count_images(val_path)
test_counts = count_images(test_path)

# Convert to DataFrame for visualization
df_counts = pd.DataFrame({
    'Train': pd.Series(train_counts),
    'Validation': pd.Series(val_counts),
    'Test': pd.Series(test_counts)
}).fillna(0).astype(int)

df_counts.plot(kind='bar', figsize=(10, 6), color=['#27ae60', '#2980b9', '#f39c12'])
plt.title('Image Count per Class')
plt.xlabel('E-Waste Category')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

 Sample Image Grid
 # Display some random images from each class in training set
def show_random_images(data_generator):
    class_indices = data_generator.class_indices
    index_to_label = {v: k for k, v in class_indices.items()}
    images, labels = next(data_generator)
    plt.figure(figsize=(14, 10))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        plt.title(index_to_label[np.argmax(labels[i])])
        plt.axis("off")
    plt.suptitle("Sample Images from Training Data", fontsize=16)
    plt.tight_layout()
    plt.show()

show_random_images(train_data)

                                                    4. Title: EfficientNetV2B0-based E-Waste Image Classification (with Fine-Tuning & Evaluation)
✅ Purpose:
This code trains an image classification model using EfficientNetV2B0 to classify e-waste images into different categories. It includes:

Data preprocessing,

Class balancing,

Two-phase training,

Evaluation using test data,

Visualization of training curves and confusion matrix.

🔍 Key Steps:
1. Setup and Configuration
Sets up image size, batch size, dataset paths, and epochs.

Uses EfficientNetV2B0 as the base model.

2. Data Preprocessing
Uses ImageDataGenerator for data augmentation in the training set.

Applies preprocess_input to normalize images.

Loads training, validation, and test images from structured directories.

3. Class Weights
Calculates balanced class weights to handle imbalanced datasets.

Weights are computed using compute_class_weight based on the training labels.

4. Model Architecture
Loads EfficientNetV2B0 with pre-trained ImageNet weights (without top).

Adds:

Global Average Pooling,

Dropout (0.3),

Dense output layer with softmax activation.

Compiles with Adam optimizer and sparse categorical cross-entropy loss.

5. Callbacks
ModelCheckpoint: Saves the best model based on validation accuracy.

EarlyStopping: Stops early if no improvement in val_loss.

ReduceLROnPlateau: Lowers learning rate if val_loss plateaus.

6. Phase 1: Train Top Layers Only
Freezes base model (feature extractor).

Trains only the top layers for 5 epochs.

7. Phase 2: Fine-Tune Entire Model
Unfreezes the base model for full fine-tuning.

Re-compiles and trains all layers for 10 more epochs.

8. Training Visualization
Plots line graphs for:

Training and validation accuracy

Training and validation loss

9. Model Evaluation on Test Data
Loads the test dataset.

Evaluates test accuracy and loss.

Prints final test accuracy.

10. Performance Metrics
Predicts test images using the model.

Computes and prints a classification report.

Plots the confusion matrix with class labels.

🧾 Outputs:
Training and validation accuracy/loss plots.

Final test accuracy (e.g., ✅ Test Accuracy: 93.42%).

Classification report with precision, recall, f1-score.

Confusion matrix heatmap.                                   
