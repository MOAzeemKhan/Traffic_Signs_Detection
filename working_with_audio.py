import os
from ultralytics import YOLO
import cv2
import pyttsx3  # For text-to-speech

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLO model
model_path = "CV_50/train/weights/best.pt"
model = YOLO(model_path)

# Open the video file (replace 'test1.mkv' with the actual path to your video file)
video_path = 'youtube-clip.mp4'
cap = cv2.VideoCapture(video_path)

# Define classes for which you want the speech output (add more as needed)
traffic_signs_with_speech = {
    0: "Arrow sign detected",
    1: "Traffic light detected",
    2: "Stop sign detected",
    3: "Speed limit detected",
    4: "Crosswalk detected"
}

# Function to handle speech output
def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    # Track detected classes
    detected_classes = set()

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # Set confidence threshold
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put the class name on the bounding box
            label = results.names[int(class_id)].upper()
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # If the class ID is in the list of traffic signs we care about, add it to detected_classes
            if int(class_id) in traffic_signs_with_speech:
                detected_classes.add(int(class_id))

    # Trigger speech output for detected classes (speak only once per frame)
    for class_id in detected_classes:
        speak(traffic_signs_with_speech[class_id])

    # Optionally, display the frame (you can comment this out if you don't want to show the frames)
    cv2.imshow("YOLO Object Detection with Speech", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Stop the text-to-speech engine
engine.stop()
