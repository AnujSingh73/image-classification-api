import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ✅ Step 1: Load Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# ✅ Step 2: Print Dataset Shape
print(f"Training Data Shape: {x_train.shape}")  # (50000, 32, 32, 3)
print(f"Testing Data Shape: {x_test.shape}")    # (10000, 32, 32, 3)

# ✅ Step 3: Display Class Labels
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print("Class Labels:", class_labels)

# ✅ Step 4: Visualize Some Images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_labels[y_train[i][0]])
    plt.axis("off")
plt.show()

# ✅ Step 5: Normalize Pixel Values (Scale 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# ✅ Step 6: Save Processed Data as NumPy Arrays
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("✅ Dataset Loading & Preprocessing Completed!")
