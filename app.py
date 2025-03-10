import os
import io
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

# 🔹 Load environment variables
load_dotenv()

app = FastAPI()

# ✅ Load API key securely from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("⚠️ API_KEY is missing! Set it in your .env file.")

# ✅ Load the trained model (Ensure it's compiled before inference)
try:
    model = tf.keras.models.load_model("image_classification_model.h5")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading model: {e}")

# ✅ Class labels (Ensure these match your dataset)
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ✅ Middleware to verify API key
async def verify_api_key(x_api_key: str = Header(...)):  
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="❌ Invalid API Key")
    return x_api_key

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key)  # API key is required before processing
):
    try:
        # ✅ Load and preprocess image
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Ensure it's always RGB
        input_size = (32, 32)  # Set to match the trained model
        image = image.resize(input_size)
        img_array = np.array(image) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # ✅ Get prediction
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        class_name = class_labels[predicted_index]

        return JSONResponse(content={
            "status": "✅ Success",
            "prediction": {
                "index": predicted_index,
                "class_name": class_name,
                "confidence": round(confidence, 4)  # Round confidence for better readability
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "❌ Error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
