import cv2
import os
import time
import numpy as np
import mediapipe as mp

from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

face_model = os.path.join(
    folder,
    "face_landmarker.task"
)

gesture_model = os.path.join(
    folder,
    "gesture_recognizer.task"
)


# ============================================================
# EMOJI FONT
# ============================================================

emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"

emoji_font = ImageFont.truetype(
    emoji_font_path,
    55
)


# ============================================================
# FACE LANDMARKER
# ============================================================

face_base = python.BaseOptions(
    model_asset_path=face_model
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    num_faces=2,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)


# ============================================================
# GESTURE RECOGNIZER
# ============================================================

gesture_base = python.BaseOptions(
    model_asset_path=gesture_model
)

gesture_options = vision.GestureRecognizerOptions(
    base_options=gesture_base,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

gesture_detector = vision.GestureRecognizer.create_from_options(
    gesture_options
)


# ============================================================
# GESTURE → EMOJI
# ============================================================

gesture_emojis = {

    "Open_Palm": "✋",

    "Closed_Fist": "✊",

    "Thumb_Up": "👍",

    "Thumb_Down": "👎",

    "Victory": "✌️",

    "Pointing_Up": "☝️",

    "ILoveYou": "🤟",

    "None": ""
}


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


timestamp = 0

previous_time = time.time()

fps = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 1


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp
    )


    face_count = len(
        face_result.face_landmarks
    )


    expression = "NEUTRAL"

    expression_emoji = "😐"


    # ========================================================
    # FACE BOX
    # ========================================================

    for face_index, face in enumerate(
        face_result.face_landmarks
    ):

        xs = [
            int(point.x * w)
            for point in face
        ]

        ys = [
            int(point.y * h)
            for point in face
        ]


        x1 = max(
            min(xs) - 15,
            0
        )

        y1 = max(
            min(ys) - 15,
            0
        )

        x2 = min(
            max(xs) + 15,
            w
        )

        y2 = min(
            max(ys) + 15,
            h
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"FACE {face_index + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ========================================================
    # EXPRESSION
    # ========================================================

    if face_result.face_blendshapes:

        blendshapes = face_result.face_blendshapes[0]

        scores = {}

        for item in blendshapes:

            scores[item.category_name] = item.score


        smile_left = scores.get(
            "mouthSmileLeft",
            0
        )

        smile_right = scores.get(
            "mouthSmileRight",
            0
        )

        jaw_open = scores.get(
            "jawOpen",
            0
        )


        if (
            smile_left > 0.4
            and smile_right > 0.4
        ):

            expression = "HAPPY"

            expression_emoji = "😄"


        elif jaw_open > 0.5:

            expression = "SURPRISED"

            expression_emoji = "😮"


        else:

            expression = "NEUTRAL"

            expression_emoji = "😐"


    # ========================================================
    # GESTURE RECOGNITION
    # ========================================================

    gesture_result = gesture_detector.recognize_for_video(
        mp_image,
        timestamp
    )


    gesture_name = "NONE"

    gesture_confidence = 0

    gesture_emoji = ""


    # ========================================================
    # PROCESS GESTURES
    # ========================================================

    if gesture_result.gestures:

        best_gesture = None

        best_score = 0

        best_hand_index = 0


        for i, gestures in enumerate(
            gesture_result.gestures
        ):

            if len(gestures) == 0:
                continue


            prediction = gestures[0]

            if prediction.score > best_score:

                best_gesture = prediction

                best_score = prediction.score

                best_hand_index = i


        if best_gesture is not None:

            gesture_name = (
                best_gesture.category_name
            )

            gesture_confidence = (
                best_gesture.score
            )

            gesture_emoji = gesture_emojis.get(
                gesture_name,
                "❓"
            )


    # ========================================================
    # DRAW HAND LANDMARKS
    # ========================================================

    if gesture_result.hand_landmarks:

        for hand in gesture_result.hand_landmarks:

            points = []

            for landmark in hand:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                points.append(
                    (x, y)
                )


            # Connections

            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 255),
                    2
                )


            # Points

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 0),
                    -1
                )


    # ========================================================
    # UI PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (620, 220),
        (15, 15, 15),
        -1
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (620, 220),
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "AI VISION SYSTEM",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FACE COUNT : {face_count}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"EXPRESSION : {expression}",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"GESTURE    : {gesture_name}",
        (25, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"CONFIDENCE : {gesture_confidence:.2f}",
        (25, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (525, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ========================================================
    # EMOJIS
    # ========================================================

    pil_image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    draw = ImageDraw.Draw(
        pil_image
    )


    draw.text(
        (540, 90),
        expression_emoji,
        font=emoji_font,
        fill="white"
    )


    if gesture_emoji:

        draw.text(
            (540, 145),
            gesture_emoji,
            font=emoji_font,
            fill="white"
        )


    # PIL → OpenCV

    frame = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Vision System - Face + Gesture",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

face_detector.close()

gesture_detector.close()