from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your MobileNet model
model = load_model(r'C:\Users\sridh\Music\DRIVER_DROWSINESS\mobilenet_drowsiness.h5')  # Replace with your model's filename

def predict_drowsiness(img_array):
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    result = "Drowsy" if np.argmax(prediction) == 1 else "Normal"
    return result, round(confidence, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Adjust size if necessary
    img_array = np.expand_dims(img / 255.0, axis=0)  # Normalize

    prediction, confidence = predict_drowsiness(img_array)

    return jsonify({"prediction": prediction, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
