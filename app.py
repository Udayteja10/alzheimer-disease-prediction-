from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import joblib
import os

app = Flask(__name__)
model, class_names = joblib.load("ml_model.pkl")

# Updated prevention tips for stages 0 through 3
prevention_tips = {
    "Stage 0": ["Maintain a healthy lifestyle", "Exercise regularly", "Eat a brain-healthy diet", "Stay mentally active"],
    "Stage 1": ["Maintain a healthy diet", "Get regular exercise", "Stay hydrated", "Stay mentally and socially active"],
    "Stage 2": ["Consult a doctor", "Avoid stress", "Take medication if prescribed", "Exercise regularly to improve cognitive function"],
    "Stage 3": ["Follow treatment plans", "Avoid exposure to triggers", "Increase rest", "Monitor cognitive function with a healthcare provider"]
}

def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((64, 64))  # Ensure the image is resized to 64x64 pixels (or based on model requirement)
    return np.array(img).flatten().reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    advice = []

    if request.method == "POST":
        file = request.files["image"]
        if file:
            processed = preprocess_image(file)
            pred = model.predict(processed)[0]
            prediction = class_names[pred]
            advice = prevention_tips.get(prediction, ["No specific advice found."])

    return render_template("index.html", prediction=prediction, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
