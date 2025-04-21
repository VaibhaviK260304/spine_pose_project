import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf

# Load the model
model = joblib.load("model/spine_pose_model.pkl")  # Make sure this model expects 132 features

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)

# Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load and preprocess image
image_path = r"C:\Users\kumbh\Desktop\spine_pose_project\dataset\normal\Untitled.jpg"
image = cv2.imread(image_path)
if image is None:
    print("❌ Image not found. Check the path:", image_path)
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (640, 640))  # Optional resize

# Show original image
cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Process image with Holistic
results = holistic.process(image_rgb)

# ========== FEATURE EXTRACTION (Pose only - 132 features) ==========
landmarks = []

if results.pose_landmarks:
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
else:
    landmarks.extend([0.0, 0.0, 0.0, 0.0] * 33)

# Convert to numpy array
keypoints = np.array(landmarks, dtype=np.float32).reshape(1, -1)

# Confirm correct number of features
expected_features = model.n_features_in_
if keypoints.shape[1] != expected_features:
    print(f"❌ Feature mismatch: got {keypoints.shape[1]} features, expected {expected_features}.")
    exit()

# ========== PREDICTION ==========
prediction = model.predict(keypoints)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(keypoints)
    print("✅ Prediction:", prediction[0])
    print("🔎 Confidence:", proba[0])
else:
    print("✅ Prediction:", prediction[0])

# ========== VISUALIZATION ==========
annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

cv2.imshow("Pose Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
