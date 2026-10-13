import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

app = FastAPI(title="Production MNIST Inference API", version="1.0.0")

# Mount the static directory so FastAPI can serve files inside it
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load ONNX Runtime session
ort_session = ort.InferenceSession("models/mnist_cnn.onnx", providers=["CPUExecutionProvider"])
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / 255.0 - 0.1307) / 0.3081
        img_array = np.expand_dims(img_array, axis=(0, 1))
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    image_bytes = await file.read()
    processed_array = preprocess_image(image_bytes)

    ort_inputs = {ort_session.get_inputs()[0].name: processed_array}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))

    return {
        "predicted_digit": predicted_class,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probabilities[0]]
    }

@app.get("/")
async def home():
    return FileResponse("app/static/index.html")