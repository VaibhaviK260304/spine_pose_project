import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        else:
            return None
