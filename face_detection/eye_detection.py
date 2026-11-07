import cv2
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MODEL
# -----------------------------

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=2,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.FaceLandmarker.create_from_options(options)


# -----------------------------
# WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

timestamp = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    # Detect face
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Number of faces
    face_count = len(result.face_landmarks)

    # Draw landmarks
    for face_landmarks in result.face_landmarks:

        h, w, _ = frame.shape

        for landmark in face_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    # -----------------------------
    # DISPLAY
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FACE COUNT: {face_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "MediaPipe Face Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()