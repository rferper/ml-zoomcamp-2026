import io
import urllib.request

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = "clothing-model.onnx"

CLASSES = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]

# PyTorch (ImageNet) normalization values
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load the model once, when the service starts
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

app = FastAPI(title="Clothing classifier")


class PredictRequest(BaseModel):
    url: str


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))


def preprocess(img: Image.Image) -> np.ndarray:
    # PyTorch (MobileNet v2): 224x224, channels first
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0   # (224, 224, 3), values 0..1
    x = (x - MEAN) / STD                           # normalize each channel
    x = x.transpose(2, 0, 1)                       # (3, 224, 224)
    return x[np.newaxis, ...].astype(np.float32)   # (1, 3, 224, 224)


def softmax(logits: np.ndarray) -> np.ndarray:
    # turns raw scores into probabilities that add up to 1
    e = np.exp(logits - logits.max())
    return e / e.sum()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    img = download_image(request.url)
    X = preprocess(img)
    logits = session.run([output_name], {input_name: X})[0][0]
    probs = softmax(logits)

    predictions = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    top_class = max(predictions, key=predictions.get)

    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=predictions[top_class],
    )