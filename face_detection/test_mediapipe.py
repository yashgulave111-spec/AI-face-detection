import os
import mediapipe as mp

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)

print("Model path:")
print(model_path)

print("\nModel exists:")
print(os.path.exists(model_path))

if os.path.exists(model_path):
    print("\nModel size:")
    print(os.path.getsize(model_path), "bytes")