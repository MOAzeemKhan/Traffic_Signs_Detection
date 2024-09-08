import os
from ultralytics import YOLO
import cv2

# Load YOLO model
model_path = "CV_50/train/weights/best.pt"
#model_path = "kosmos_dataset_azeem-phone/best_onlyright.pt"
model = YOLO(model_path)

# Open the video file (replace 'your_video_path.mp4' with the actual path to your video file)
video_path = 'test1.mkv'
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

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

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
