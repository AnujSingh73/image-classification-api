import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# ✅ Step 1: Load Preprocessed Data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

# ✅ Step 2: Normalize Data (Improves Training)
x_train = x_train / 255.0

# ✅ Step 3: Define Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=15,      # Rotate images by 15 degrees
    width_shift_range=0.1,  # Shift width by 10%
    height_shift_range=0.1, # Shift height by 10%
    horizontal_flip=True,   # Flip images horizontally
    zoom_range=0.1          # Zoom images by 10%
)

# ✅ Step 4: Apply Data Augmentation & Preview 5 Transformed Images
augmented_images = data_gen.flow(x_train, y_train, batch_size=5, shuffle=True)

plt.figure(figsize=(10, 5))
for i in range(5):
    batch = next(augmented_images)  # Get a batch
    img, label = batch[0], batch[1]  # Extract first image & label
    plt.subplot(1, 5, i + 1)
    plt.imshow(img[0])  # Display first image in batch
    plt.axis("off")
plt.show()

print("✅ Data Augmentation Applied Successfully!")
