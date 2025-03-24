import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Set correct project path
project_path = r"C:\Users\rekre\OneDrive\Desktop\chest_xray"

# Define dataset paths
test_path = os.path.join(project_path, "test")

# Load the trained model
model = load_model("chest_xray_model.h5")
print("Model loaded successfully!")

# Hyperparameters
hyper_dimension = 224  # Adjusted to match model input shape
hyper_batch_size = 32
hyper_channels = 1
hyper_mode = 'grayscale'  # Using grayscale since it's medical images

# Data Augmentation for Test Data
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load test dataset
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(hyper_dimension, hyper_dimension),  # Ensure 224x224 input size
    batch_size=hyper_batch_size,
    color_mode=hyper_mode,
    class_mode='binary',
    seed=42
)

# 1. Plot confusion matrix
def plot_confusion_matrix(model, test_generator):
    # Predict the test dataset in batches
    y_true = test_generator.classes
    y_pred = []
    
    # Loop through all batches in the generator
    for i in range(len(test_generator)):
        batch_images, _ = test_generator[i]
        batch_preds = model.predict(batch_images)
        y_pred.extend(batch_preds)

    # Convert predictions to binary (0 or 1)
    y_pred_classes = (np.array(y_pred) > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# 2. Plot predictions vs actuals
def plot_predictions_vs_actuals(model, test_generator):
    # Predict the test dataset in batches
    y_true = test_generator.classes
    y_pred = []
    
    # Loop through all batches in the generator
    for i in range(len(test_generator)):
        batch_images, _ = test_generator[i]
        batch_preds = model.predict(batch_images)
        y_pred.extend(batch_preds)

    # Convert predictions to binary (0 or 1)
    y_pred_classes = (np.array(y_pred) > 0.5).astype(int)

    # Plot the first 10 predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:10], label='Actual', marker='o')
    plt.plot(y_pred_classes[:10], label='Predicted', marker='x')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Test Samples')
    plt.ylabel('Prediction (Normal/Pneumonia)')
    plt.xticks(ticks=np.arange(10), labels=[f"Sample {i+1}" for i in range(10)])
    plt.legend()
    plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test set
predictions = model.predict(test_generator)
predictions = (predictions > 0.5)  # Convert to binary labels (0 or 1)

# Print predictions
print("\nPredictions on test data:")
print(predictions)

# Plot the graphs
plot_confusion_matrix(model, test_generator)  # Plot confusion matrix
plot_predictions_vs_actuals(model, test_generator) 
