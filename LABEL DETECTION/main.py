from ultralytics import YOLO
import cv2
import math

# Model
model = YOLO("/LABEL DETECTION/weights/best.pt")

# Start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform inference
    results = model(frame)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Check confidence threshold
            if confidence >= 0.8:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                print("Confidence --->", confidence)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
