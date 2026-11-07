import cv2
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# MODEL PATH
# ==========================================

folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    folder,
    "gesture_recognizer.task"
)


# ==========================================
# CREATE GESTURE RECOGNIZER
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

recognizer = vision.GestureRecognizer.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

timestamp = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 1


    # ======================================
    # AI GESTURE RECOGNITION
    # ======================================

    result = recognizer.recognize_for_video(
        mp_image,
        timestamp
    )


    # ======================================
    # PROCESS RESULTS
    # ======================================

    if result.gestures:

        for i, gestures in enumerate(result.gestures):

            if len(gestures) == 0:
                continue


            # Best prediction
            best = gestures[0]

            gesture_name = best.category_name

            confidence = best.score


            # --------------------------------
            # Display gesture
            # --------------------------------

            cv2.putText(
                frame,
                f"Gesture: {gesture_name}",
                (30, 50 + i * 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Display confidence
            # --------------------------------

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (30, 80 + i * 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


    else:

        cv2.putText(
            frame,
            "Gesture: NONE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "MediaPipe AI Gesture Recognition",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

recognizer.close()