import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split

# ✅ Step 1: Load Preprocessed Data
x_train = np.load("x_train.npy")  # Ensure file exists
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# ✅ Step 2: Normalize the Data
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

# ✅ Step 3: Train-Validation Split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(f"Train Shape: {x_train.shape}, Validation Shape: {x_val.shape}, Test Shape: {x_test.shape}")

# ✅ Step 4: Build a CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)),  # Adjusted input size
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),  # ✅ Prevents overfitting
    layers.Dense(10, activation="softmax")  # 10 output classes
])

# ✅ Step 5: Compile the Model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ✅ Step 6: Train the Model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=25,  # Increased epochs for better learning
    batch_size=32
)

# ✅ Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ✅ Step 8: Save the Model
model.save("image_classification_model.h5")

print("\n✅ Model Training Completed & Saved!")
