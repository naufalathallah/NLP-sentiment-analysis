from google.colab import drive
import os
import zipfile
import shutil
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, BatchNormalization

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
zip_path = '/content/drive/MyDrive/Dicoding/dataset-img/Archive.zip'
dataset_path = '/content/dataset/'
resized_dataset_path = '/content/dataset_resized_128x128/' 

# Clean up existing directories
if os.path.exists(resized_dataset_path):
    shutil.rmtree(resized_dataset_path)
    print(f"🔥 Folder {resized_dataset_path} telah dihapus untuk memastikan clean start.")

if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)
    print(f"🔥 Folder {dataset_path} telah dihapus untuk memastikan clean start.")

# Extract dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

# Remove unwanted folders
unwanted_folders = ['__MACOSX'] + [f for f in os.listdir(dataset_path) if f.startswith('_resized_')]
for folder in unwanted_folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🔥 Folder {folder} telah dihapus.")

print("✅ Dataset berhasil diekstrak ke:", dataset_path)
print("📂 Daftar folder dalam dataset setelah pembersihan:", os.listdir(dataset_path))

# Count images per category
class_counts = {}
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        num_files = len(os.listdir(category_path))
        class_counts[category] = num_files

total_images = sum(class_counts.values())
print("Jumlah gambar per kategori:")
for category, count in class_counts.items():
    print(f"{category}: {count} images")
print(f"\nTotal jumlah gambar dalam dataset: {total_images} images")

# Resize images to 128x128
target_size = (128, 128)
dataset_path = dataset_path.rstrip('/')
resized_dir = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_resized_{target_size[0]}x{target_size[1]}")
os.makedirs(resized_dir, exist_ok=True)

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        save_category_path = os.path.join(resized_dir, category)
        os.makedirs(save_category_path, exist_ok=True)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                save_path = os.path.join(save_category_path, img_name)
                cv2.imwrite(save_path, img_resized)

print(f"✅ Dataset berhasil diresize ke {target_size} dan disimpan di {resized_dir}")

# Split dataset into train, validation, and test sets
split_base_dir = "/content/dataset_split"
train_dir = os.path.join(split_base_dir, "train")
val_dir = os.path.join(split_base_dir, "val")
test_dir = os.path.join(split_base_dir, "test")

if os.path.exists(split_base_dir):
    shutil.rmtree(split_base_dir)
    print(f"🔥 Folder {split_base_dir} telah dihapus untuk memastikan clean start.")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create category folders in train, val, test directories
for category in os.listdir(resized_dataset_path):
    category_path = os.path.join(resized_dataset_path, category)
    
    if os.path.isdir(category_path):
        # Create category folders in each split directory
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
        # Get all image filenames
        all_files = os.listdir(category_path)
        
        # Split files into train, val, test
        train_files, temp_files = train_test_split(all_files, test_size=(1-train_ratio), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
        
        # Copy files to respective directories
        for file_name in train_files:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(train_dir, category, file_name)
            shutil.copy(src, dst)
            
        for file_name in val_files:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(val_dir, category, file_name)
            shutil.copy(src, dst)
            
        for file_name in test_files:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(test_dir, category, file_name)
            shutil.copy(src, dst)
        
        print(f"Kategori {category}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

print("✅ Dataset berhasil dibagi menjadi Train, Validation, dan Test set.")

# Data augmentation with enhanced transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Get number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {train_generator.class_indices}")

# Build improved CNN model with batch normalization and regularization
model = Sequential([
    # First convolutional block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3),
                 kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Learning rate and optimizer
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

# Train the model (uncomment to train)
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     callbacks=callbacks,
#     verbose=1
# )