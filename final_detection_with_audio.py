import os
from ultralytics import YOLO
import cv2
import pyttsx3
import moviepy.editor as mpe
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Function to recognize speed limit from cropped image using Tesseract
def recognize_speed_limit(cropped_image):
    # Convert image to grayscale (for better OCR results)
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to extract text
    config = '--psm 6 digits'
    speed_limit_text = pytesseract.image_to_string(gray_image, config=config)

    # Extract only digits (numbers) as speed limit
    speed_limit = ''.join(filter(str.isdigit, speed_limit_text))

    return speed_limit

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLO model
model_path = "CV_50/train/weights/best.pt"
model = YOLO(model_path)
sign_model = YOLO("Traffic_Sign_Model/train/weights/best.pt")

# Open the video file (replace 'test1.mkv' with the actual path to your video file)
video_path = 'test_traffic_sign_vid.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video (without audio for now)
output_video_path = 'output_video.avi'  # Change the output path as needed
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use other codecs like 'mp4v' for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create the extracted_frames directory if it doesn't exist
extracted_frames_dir = "extracted_frames"
if not os.path.exists(extracted_frames_dir):
    os.makedirs(extracted_frames_dir)

# Directory for temporary audio files
if not os.path.exists("audio"):
    os.makedirs("audio")

# Initialize previous detected object to avoid repeated speech
previously_detected = None

# Counter for naming audio files and saved frames
audio_counter = 0
frame_counter = 0
audio_clips = []

# Define a speech function that saves audio
def save_speech(text, audio_counter):
    audio_file = f"audio/speech_{audio_counter}.mp3"
    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    return audio_file

# Start video processing
print("Video processing started. Press 'q' to stop.")
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    # Track detected classes in the current frame
    detected_classes = []

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # Set confidence threshold
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put the class name on the bounding box
            class_name = results.names[int(class_id)].upper()
            cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cropped_bounding_box = frame[int(y1):int(y2), int(x1):int(x2)]

            # If a new class is detected, save the speech
            if class_name != previously_detected:
                if class_name == "ARROW":
                    audio_file = save_speech("Arrow detected", audio_counter)
                    audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                elif class_name == "TRAFFIC LIGHT":
                    sign_results = sign_model(cropped_bounding_box)[0]
                    #print(sign_results)
                    # Process the output from the second model (sign_model)
                    for sign_result in sign_results.boxes.data.tolist():
                        # Process the bounding box from the second model
                        sign_x1, sign_y1, sign_x2, sign_y2, sign_score, sign_class_id = sign_result

                        if sign_score > 0.5:  # Set a confidence threshold
                            # Identify the color or sign based on the sign_class_id
                            if sign_class_id == 0:
                                detected_color = "Green"
                            elif sign_class_id == 1:
                                detected_color = "Red"
                            elif sign_class_id == 2:
                                detected_color = "Yellow"

                            # Write the detected color above the bounding box
                            cv2.putText(frame, f"Traffic Light: {detected_color}", (int(x1), int(y1 - 25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                            #print(detected_color)
                            # Generate speech for the detected color
                            audio_file = save_speech(f"Traffic light detected: {detected_color}", audio_counter)
                            audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                elif class_name == "STOP SIGN":
                    audio_file = save_speech("Stop sign detected", audio_counter)
                    audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                elif class_name == "SPEED LIMIT":
                    # Recognize the speed limit using OCR
                    speed_limit = recognize_speed_limit(cropped_bounding_box)
                    # If a valid speed limit is recognized, display it and save the audio
                    if speed_limit:
                        print("SPEED LIMIT IDENTIFIED:", speed_limit)
                        # Update the bounding box label with the recognized speed limit
                        cv2.putText(frame, f"Speed Limit: {speed_limit} km/h", (int(x1), int(y1 - 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        # Generate speech for the recognized speed limit
                        audio_file = save_speech(f"Speed limit detected: {speed_limit} kilometers per hour",
                                                 audio_counter)
                        audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                    else:
                        print("No speed limit detected")
                        # If OCR fails, fallback to the generic speech
                        audio_file = save_speech("Speed limit detected", audio_counter)
                        audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                elif class_name == "CROSSWALK":
                    audio_file = save_speech("Crosswalk detected", audio_counter)
                    audio_clips.append((audio_file, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))

                audio_counter += 1
                previously_detected = class_name

            # Save the cropped bounding box to the extracted_frames directory instead of the full frame
            bounding_box_filename = os.path.join(extracted_frames_dir, f"bbox_{frame_counter}.jpg")

            # Crop the detected object and save it
            cv2.imwrite(bounding_box_filename, cropped_bounding_box)
            frame_counter += 1

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Optionally, display the frame (you can comment this out if you don't want to show the frames)
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

    # List of audio clips with start times
    combined_audio = []
    for audio_file, start_time in audio_clips:
        audio_clip = mpe.AudioFileClip(audio_file).set_start(start_time)
        combined_audio.append(audio_clip)

    # Combine all audio into a single audio file
    final_audio = mpe.CompositeAudioClip(combined_audio)

    # Set the final audio to the video
    final_video = video_clip.set_audio(final_audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Final output with audio
final_output_video_path = "final_output_with_audio_sign.mp4"
combine_video_audio(output_video_path, audio_clips, final_output_video_path)

# Print output video path
print(f"Final video saved with speech audio as {final_output_video_path}")
