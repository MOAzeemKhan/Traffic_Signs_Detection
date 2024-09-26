import os
from ultralytics import YOLO
import cv2
import pyttsx3
import moviepy.editor as mpe
import numpy as np
import easyocr

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLO models
model_path = "CV_50/train/weights/best.pt"
model = YOLO(model_path)
sign_model = YOLO("Traffic_Sign_Model/train/weights/best.pt")

video_path = 'daivik_test.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if fps is zero

# Define the codec and create a VideoWriter object to save the output video (without audio for now)
output_video_path = 'output_video_daivik_test.avi'  # Change the output path as needed
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use other codecs like 'mp4v' for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create directories for temporary audio files
audio_dir = "audio"
os.makedirs(audio_dir, exist_ok=True)

# Initialize previous detected object to avoid repeated speech
previously_detected = None

# Counter for naming audio files
audio_counter = 0
audio_clips = []

# Define a speech function that saves audio
def save_speech(text, audio_counter):
    audio_file = os.path.join(audio_dir, f"speech_{audio_counter}.mp3")
    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    return audio_file

# Start video processing
print("Video processing started.")
frame_count = 0
frame_skip_interval = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip_interval != 0:
        out.write(frame)  # Write original frame if skipping processing
        continue

    # Optionally resize frame for faster processing
    #frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # Perform object detection
    results = model(frame)[0]

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # Set confidence threshold
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the class name
            class_name = results.names[int(class_id)].upper()

            # Crop the detected object
            cropped_bounding_box = frame[y1:y2, x1:x2]

            # Initialize variables for detected attributes
            speed_limit = None
            detected_color = None

            # Process based on class name
            if class_name == "SPEED LIMIT":
                # Recognize the speed limit using OCR
                ocr_results = reader.readtext(cropped_bounding_box)
                for detection in ocr_results:
                    speed_limit = detection[1]
                    if speed_limit:
                        print("SPEED LIMIT IDENTIFIED:", speed_limit)
                        break
                label = f"Speed Limit: {speed_limit or 'Not detected'} km/h"
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, cv2.LINE_AA)

            elif class_name == "ARROW":
                # Placeholder for direction prediction
                direction = "Unknown"
                print("ARROW IDENTIFIED:", direction)
                cv2.putText(frame, f"Direction: {direction}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            elif class_name == "TRAFFIC LIGHT":
                # Detect traffic light color using the secondary model
                sign_results = sign_model(cropped_bounding_box)[0]
                for sign_result in sign_results.boxes.data.tolist():
                    sign_score, sign_class_id = sign_result[4], int(sign_result[5])
                    if sign_score > 0.5:
                        detected_color = ["Green", "Red", "Yellow"][sign_class_id]
                        print("TRAFFIC LIGHT IDENTIFIED:", detected_color)
                        break
                label = f"Traffic Light: {detected_color or 'Unknown'}"
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Generate speech if a new class is detected
            if class_name != previously_detected:
                frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
                if class_name == "SPEED LIMIT":
                    text = f"Speed limit detected: {speed_limit} kilometers per hour" if speed_limit else "Speed limit detected"
                elif class_name == "TRAFFIC LIGHT" and detected_color:
                    text = f"Traffic light detected: {detected_color}"
                elif class_name == "ARROW":
                    text = "Arrow detected"
                elif class_name == "STOP SIGN":
                    text = "Stop sign detected"
                elif class_name == "CROSSWALK":
                    text = "Crosswalk detected"
                else:
                    text = f"{class_name.replace('_', ' ').title()} detected"

                # Save the speech audio
                audio_file = save_speech(text, audio_counter)
                audio_clips.append((audio_file, frame_time))
                audio_counter += 1
                previously_detected = class_name

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("YOLO Object Detection with Speech", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects, and close the OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Combine video with audio clips
def combine_video_audio(video_path, audio_clips, output_path):
    video_clip = mpe.VideoFileClip(video_path)

    # Create audio clips with start times
    combined_audio = []
    for audio_file, start_time in audio_clips:
        audio_clip = mpe.AudioFileClip(audio_file).set_start(start_time)
        combined_audio.append(audio_clip)

    # Combine all audio clips
    if combined_audio:
        final_audio = mpe.CompositeAudioClip(combined_audio)
        final_video = video_clip.set_audio(final_audio)
    else:
        final_video = video_clip

    # Write the final video file
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Final output with audio
final_output_video_path = "final_output_with_audio_sign.mp4"
combine_video_audio(output_video_path, audio_clips, final_output_video_path)

# Print output video path
print(f"Final video saved with speech audio as {final_output_video_path}")
