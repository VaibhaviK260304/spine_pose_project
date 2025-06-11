import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from collections import Counter

# Paths
DATA_DIR = 'dataset'  # contains subfolders: scoliosis/, normal/
LABELS = {'normal': 0, 'scoliosis': 1}

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark
    # Use hip center as reference point
    try:
        hip_center = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    except IndexError:
        return None

    features = []
    for lm in landmarks:
        features.extend([
            lm.x - hip_center.x,
            lm.y - hip_center.y,
            lm.z - hip_center.z
        ])
    return features

# Load dataset
X, y = [], []

for label_name in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(label_path):
        continue

    for img_file in os.listdir(label_path):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(label_path, img_file)
        features = extract_landmarks(img_path)
        if features:
            X.append(features)
            y.append(LABELS[label_name])

print(f"✅ Total samples: {len(X)}")
print(f"📊 Label counts: {Counter(y)}")

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📈 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, 'spine_pose_model_new.pkl')
print("✅ Model saved as spine_pose_model.pkl")
