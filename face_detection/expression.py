import cv2
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# LOAD FACE LANDMARK MODEL
# ==========================================

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


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    face_count = len(result.face_landmarks)

    expression = "😐 NEUTRAL"

    # ==========================================
    # EXPRESSION DETECTION
    # ==========================================

    if result.face_blendshapes:

        blendshapes = result.face_blendshapes[0]

        scores = {}

        for item in blendshapes:
            scores[item.category_name] = item.score

        # Smile detection
        smile_left = scores.get(
            "mouthSmileLeft",
            0
        )

        smile_right = scores.get(
            "mouthSmileRight",
            0
        )

        # Mouth open detection
        jaw_open = scores.get(
            "jawOpen",
            0
        )

        # Determine expression

        if smile_left > 0.4 and smile_right > 0.4:

            expression = "😄 HAPPY"

        elif jaw_open > 0.5:

            expression = "😮 SURPRISED"

        else:

            expression = "😐 NEUTRAL"


    # ==========================================
    # DISPLAY
    # ==========================================

    # Face count box

    cv2.rectangle(
        frame,
        (10, 10),
        (450, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FACE COUNT: {face_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Expression
    cv2.putText(
        frame,
        f"EXPRESSION: {expression}",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow(
        "AI Face Expression",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLOSE
# ==========================================

cap.release()
cv2.destroyAllWindows()

detector.close()