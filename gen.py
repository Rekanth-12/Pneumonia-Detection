import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set correct project path
project_path = r"C:\Users\rekre\OneDrive\Desktop\chest_xray"

# Define dataset paths
train_path = os.path.join(project_path, "train")
val_path = os.path.join(project_path, "val")
test_path = os.path.join(project_path, "test")

# Print paths for verification
print(f"Train path: {train_path}")
print(f"Val path: {val_path}")
print(f"Test path: {test_path}")

# Check if dataset directories exist
for path in [train_path, val_path, test_path]:
    if not os.path.exists(path):
        print(f"❌ Error: Directory does not exist -> {path}")
        exit()

# Hyperparameters
hyper_dimension = 224  # Adjusted to match model input shape
hyper_batch_size = 32
hyper_epochs = 100
hyper_channels = 1
hyper_mode = 'grayscale'  # Using grayscale since it's medical images

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(hyper_dimension, hyper_dimension),  # Ensure 224x224 input size
    batch_size=hyper_batch_size,
    color_mode=hyper_mode,
    class_mode='binary',
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(hyper_dimension, hyper_dimension),  # Ensure 224x224 input size
    batch_size=hyper_batch_size,
    color_mode=hyper_mode,
    class_mode='binary',
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(hyper_dimension, hyper_dimension),  # Ensure 224x224 input size
    batch_size=hyper_batch_size,
    color_mode=hyper_mode,
    class_mode='binary',
    seed=42
)

# Print dataset details
print("\n✅ Data loaded successfully!")
print(f"Train dataset: {train_generator.samples} samples")
print(f"Validation dataset: {val_generator.samples} samples")
print(f"Test dataset: {test_generator.samples} samples")
print(f"Classes: {train_generator.class_indices}")
