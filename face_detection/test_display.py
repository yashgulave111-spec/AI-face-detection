print("PROGRAM STARTED")
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # BIG RED rectangle
    cv2.rectangle(
        frame,
        (10, 10),
        (500, 100),
        (0, 0, 255),
        -1
    )

    # WHITE text
    cv2.putText(
        frame,
        "HELLO CAMERA",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    cv2.imshow("TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()