from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 Add this line
import numpy as np
import cv2
import mediapipe as mp
import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('spine_pose_model.pkl')

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Temp save path
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    predictions = []

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        features = extract_landmarks(file_path)
        if features:
            pred = model.predict([features])[0]
            confidence = model.predict_proba([features])[0].tolist()
            predictions.append({'filename': filename, 'prediction': int(pred), 'confidence': confidence})
        else:
            predictions.append({'filename': filename, 'error': 'Could not detect pose'})

        os.remove(file_path)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
