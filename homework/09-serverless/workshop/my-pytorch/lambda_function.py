import io
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession(
    "clothing_classifier_mobilenet_v2_latest.onnx",
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
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


def download_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return Image.open(io.BytesIO(resp.read()))


def preprocess(img):
    # PyTorch (MobileNet v2) model: 224x224 images, channels first
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0   # (224, 224, 3), values 0..1
    x = (x - MEAN) / STD                           # normalize each channel
    x = x.transpose(2, 0, 1)                       # (3, 224, 224): channels first
    return x[np.newaxis, ...].astype(np.float32)   # (1, 3, 224, 224): add batch


def predict(url):
    img = download_image(url)
    X = preprocess(img)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result