# image-classification-api
 An assignment by SOULAI : Image Classification Model
# Image Classification API using FastAPI & Docker

## 📌 Project Overview

This project is an end-to-end **Image Classification API** that utilizes a Convolutional Neural Network (CNN) for classifying images. The API is built using **FastAPI**, containerized with **Docker**, and supports deployment on cloud platforms.

## 📁 Project Structure

```
📂 image-classification-api/
│-- 📂 notebooks/                # Jupyter Notebooks for training & testing
│-- 📂 src/                      # Source code
│   │-- model.py                 # CNN model definition
│   │-- train_model.py           # Model training script
│   │-- preprocess.py            # Data preprocessing script
│   │-- dataset_loading.py       # Dataset loading script
│   │-- data_augmentation.py     # Data augmentation techniques
│-- 📂 api/                      # API Code
│   │-- app.py                   # FastAPI script
│-- 📂 docker/                   # Docker-related files
│   │-- Dockerfile               # Dockerfile to containerize API
│   │-- requirements.txt         # Dependencies
│-- 📂 models/                   # Trained model files
│   │-- model.pth (PyTorch) or model.h5 (TensorFlow)
│   │-- model.pkl (if using sklearn)
│   │-- model.onnx (if using ONNX format)
│-- report.md                    # Report on model training & deployment
│-- README.md                    # Project Overview & Setup Instructions
│-- requirements.txt              # Dependencies for the project
│-- docker-compose.yml (optional) # If using Docker Compose
```

---

## ✅ Steps to Complete the Project

| Step | Task                                     | Status |
| ---- | ---------------------------------------- | ------ |
| 1    | Data Preprocessing & Feature Engineering | ✅ Done |
| 2    | Model Selection & Training               | ✅ Done |
| 3    | Hyperparameter Tuning                    | ✅ Done |
| 4    | API Development (FastAPI)                | ✅ Done |
| 5    | Dockerization                            | ✅ Done |
| 6    | Deployment & API Testing                 | ✅ Done |
| 7    | Documentation (README & Report)          | ✅ Done |

---

## 🚀 How to Run the Project

### 🔹 **1. Clone the Repository**

```sh
git clone https://github.com/AnujSingh73/image-classification-api.git
cd image-classification-api
```

### 🔹 **2. Install Dependencies**

```sh
pip install -r requirements.txt
```

### 🔹 **3. Run Data Preprocessing & Model Training**

```sh
python src/preprocess.py
python src/dataset_loading.py
python src/data_augmentation.py
python src/train_model.py
```

### 🔹 **4. Save the Trained Model**

Depending on the framework used, save the model in the desired format:

**PyTorch (.pth):**

```python
import torch
torch.save(model.state_dict(), "models/model.pth")
```

**TensorFlow/Keras (.h5):**

```python
model.save("models/model.h5")
```

**ONNX format:**

```python
import torch
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on model input
torch.onnx.export(model, dummy_input, "models/model.onnx")
```

**Scikit-learn (.pkl):**

```python
import pickle
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### 🔹 **5. Run FastAPI Server**

```sh
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://127.0.0.1:8000/docs` (Swagger UI).

### 🔹 **6. Run API with Docker**

```sh
# Build the Docker Image
docker build -t image-classification-api .

# Run the Docker Container
docker run -d -p 8000:8000 --name fastapi-container image-classification-api
```

### 🔹 **7. Run API using Postman**

1. Open **Postman**.
2. Select `POST` method.
3. Enter URL: `http://127.0.0.1:8000/predict`
4. Go to `Body` → `form-data` → Upload an image.
5. Click `Send` to get classification results.

---

## 📡 API Endpoints

| Endpoint   | Method | Description                                 |
| ---------- | ------ | ------------------------------------------- |
| `/predict` | `POST` | Upload an image & get classification result |
| `/health`  | `GET`  | Check if API is running                     |

---

## 📜 About

### 🔹 **Steps Taken for Data Preprocessing & Feature Engineering**

- Loaded dataset using `dataset_loading.py`.
- Applied normalization, resizing, and augmentation using `data_augmentation.py`.
- Split dataset into training, validation, and test sets.

### 🔹 **Model Selection & Optimization Approach**

- Used **CNN (Convolutional Neural Network)** for image classification.
- Experimented with different architectures (ResNet, VGG, Custom CNN).
- Applied **Hyperparameter tuning** using GridSearchCV & KerasTuner.
- Optimized using **Adam optimizer** and **Cross-entropy loss**.

### 🔹 **Deployment Strategy & API Usage Guide**

- Developed an API using **FastAPI**.
- Containerized API using **Docker**.
- Deployed on **Render** (optional: AWS/GCP/Azure).
- API accessible via `http://127.0.0.1:8000/docs`.
- API tested using **Postman & Curl**.

---

## 📌 Conclusion

This project successfully builds an **image classification API** that can process images, predict labels, and provide results via **FastAPI endpoints**. The model is **trained, optimized, and deployed** using Docker for seamless execution.

---

### 🔗 **GitHub Repository:** [GitHub Link](https://github.com/AnujSingh73/image-classification-api)

---

🚀 **For any issues, feel free to raise a GitHub issue or contribute!**

