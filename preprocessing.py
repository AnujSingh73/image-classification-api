import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load CIFAR-10 Dataset (Modify if using a different dataset)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# ✅ Normalize pixel values (Scale between 0 and 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# ✅ Flatten labels (Convert from (num_samples, 1) → (num_samples,))
y_train, y_test = y_train.flatten(), y_test.flatten()

# ✅ Split training data into training (80%) and validation (20%)
split_index = int(0.8 * len(x_train))  # 80% of data
x_train, x_val = x_train[:split_index], x_train[split_index:]
y_train, y_val = y_train[:split_index], y_train[split_index:]

# ✅ Define Data Augmentation (For Training Pipeline)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images
    tf.keras.layers.RandomRotation(0.1),       # Rotate images by 10%
    tf.keras.layers.RandomZoom(0.1),           # Zoom images slightly
])

# ✅ Visualize Augmented Images
plt.figure(figsize=(10, 5))
for i in range(5):
    img_tensor = tf.expand_dims(x_train[i], axis=0)  # Expand batch dimension
    augmented_img = data_augmentation(img_tensor)    # Apply augmentation
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmented_img[0].numpy())  # Convert back to NumPy
    plt.axis("off")
plt.show()

# ✅ Convert to NumPy Arrays for Model Training
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val)
np.save("y_val.npy", y_val)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("✅ Data Preprocessing Completed & Saved!")
