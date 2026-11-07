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
    "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

timestamp = 0


# -----------------------------
# GESTURE FUNCTION
# -----------------------------
def get_gesture(hand):

    # Landmark numbers:
    # Thumb tip = 4
    # Index tip = 8
    # Middle tip = 12
    # Ring tip = 16
    # Pinky tip = 20

    # PIP joints
    # Index = 6
    # Middle = 10
    # Ring = 14
    # Pinky = 18

    fingers = []

    # Index
    fingers.append(hand[8].y < hand[6].y)

    # Middle
    fingers.append(hand[12].y < hand[10].y)

    # Ring
    fingers.append(hand[16].y < hand[14].y)

    # Pinky
    fingers.append(hand[20].y < hand[18].y)

    # Thumb
    fingers.append(abs(hand[4].x - hand[3].x) > 0.04)

    # -----------------------------
    # GESTURE CLASSIFICATION
    # -----------------------------

    # Open palm
    if all(fingers):
        return "OPEN PALM"

    # Fist
    if not any(fingers):
        return "FIST"

    # Peace / Victory
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "VICTORY"

    # Pointing
    if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
        return "POINTING"

    # Thumb up
    if fingers[4] and not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
        return "THUMBS UP"

    return "UNKNOWN"


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    hand_count = len(result.hand_landmarks)

    # Display hand count
    cv2.rectangle(
        frame,
        (10, 10),
        (430, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"HAND COUNT: {hand_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # PROCESS HANDS
    # -----------------------------
    for i, hand in enumerate(result.hand_landmarks):

        gesture = get_gesture(hand)

        # Convert normalized coordinates to pixels
        h, w, _ = frame.shape

        x = int(hand[0].x * w)
        y = int(hand[0].y * h)

        # Display gesture
        cv2.putText(
            frame,
            gesture,
            (x - 50, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Draw hand landmarks
        for landmark in hand:

            lx = int(landmark.x * w)
            ly = int(landmark.y * h)

            cv2.circle(
                frame,
                (lx, ly),
                4,
                (0, 255, 0),
                -1
            )

    cv2.imshow(
        "AI Hand Gesture Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()