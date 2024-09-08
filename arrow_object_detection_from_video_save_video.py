import os
from ultralytics import YOLO
import cv2

# Load YOLO model
model_path = "CV_50/train/weights/best.pt"
model = YOLO(model_path)

# Open the video file (replace 'test1.mkv' with the actual path to your video file)
video_path = 'youtube-clip.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output_video_path = 'output_video.avi'  # Change the output path as needed
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use other codecs like 'mp4v' for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print("Video processing started. Press 'q' to stop.")
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

        if score > 0.5:  # Set confidence threshold
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put the class name on the bounding box
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Optionally, display the frame (you can comment this out if you don't want to show the frames)
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects, and close the OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Print output video path
print(f"Video saved as {output_video_path}")
