import os
from ultralytics import YOLO
import cv2

# Load YOLO model
model_path = "best2.pt"
model = YOLO(model_path)

# Open the camera
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("/dev/video1")
# Set the frame dimensions (adjust based on your camera)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)[0]

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
