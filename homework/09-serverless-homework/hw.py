from io import BytesIO
from urllib import request

import numpy as np
import onnxruntime as ort
from PIL import Image

SAMPLE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def download_image(url):
    with request.urlopen(url, timeout=30) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def prepare_image(image):
    image = image.resize((200, 200), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))[None, ...]


session = ort.InferenceSession(
    "hair_classifier_v1.onnx",
    providers=["CPUExecutionProvider"],
)

# Question 1: input and output names
print("INPUTS:")
for inp in session.get_inputs():
    print("  name:", inp.name, "| shape:", inp.shape)
print("OUTPUTS:")
for out in session.get_outputs():
    print("  name:", out.name, "| shape:", out.shape)

# Question 3: first value of the R channel
X = prepare_image(download_image(SAMPLE_URL))
print("\nTensor shape:", X.shape)
print("Q3 - tensor[0, 0, 0, 0]:", round(float(X[0, 0, 0, 0]), 3))

# Question 4: model output
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: X})[0]
print("Q4 - straight probability:", round(float(output.ravel()[0]), 3))