import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import mediapipe as mp
DATASET_PATH = r"C:\Users\kumbh\Desktop\spine_pose_project\dataset"

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

DATASET_PATH = r"C:\Users\kumbh\Desktop\spine_pose_project\dataset"

X, y = [], []

def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Flatten the landmark coordinates into a single vector
        return np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in landmarks]).flatten()
    else:
        return None

# Load dataset and extract features
for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        landmarks = extract_pose_landmarks(img_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(label)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/spine_pose_model.pkl")
print("✅ Model trained and saved.")
