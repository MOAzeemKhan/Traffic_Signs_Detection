# Video Object Detection with Audio Alerts

This project implements real-time object detection on video frames using YOLO models and provides audio alerts for detected objects. It processes a video file, identifies specific traffic-related signs and signals, generates corresponding speech outputs, and produces a final video with bounding boxes and synchronized audio alerts.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Functionality Overview](#functionality-overview)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Object Detection**: Uses YOLO models to detect objects in video frames.
- **Traffic Sign Recognition**: Identifies traffic signs like speed limits, arrows, traffic lights, stop signs, and crosswalks.
- **Optical Character Recognition (OCR)**: Reads speed limit values from signs using EasyOCR.
- **Audio Alerts**: Generates speech alerts for detected objects using a text-to-speech engine.
- **Video Output**: Creates a video file with bounding boxes and labels overlaid, along with synchronized audio alerts.

## Prerequisites

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) with CUDA support (if using GPU acceleration)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [moviepy](https://pypi.org/project/moviepy/)
- [numpy](https://numpy.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [torchvision](https://pypi.org/project/torchvision/) (for model loading)
- [torch](https://pypi.org/project/torch/) (for model loading)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Since `requirements.txt` is not provided, you can install the dependencies individually:

   ```bash
   pip install ultralytics opencv-python pyttsx3 moviepy numpy easyocr torch torchvision
   ```

4. **Download Custom YOLO Model**

   - **Primary Object Detection Model**

     Download a YOLOv8 model (e.g., `yolov8n.pt`) and place it in the project directory.

   - **Secondary Model for Traffic Light Detection**

     If using a custom model for traffic light color detection, ensure it's downloaded and update the path in the code.

## Usage

1. **Prepare the Video File**

   Place your input video file in the project directory and update the `video_path` variable in the code if necessary.

2. **Run the Script**

   ```bash
   python your_script_name.py
   ```

3. **Output**

   - The script will process the video, detect objects, generate audio alerts, and produce a final video file named `final_output_with_audio_sign.mp4` in the project directory.
   - Temporary audio files will be saved in the `audio` directory.

## Project Structure

```
your_repository/
├── your_script_name.py
├── requirements.txt
├── final_output_with_audio_sign.mp4
├── audio/
│   ├── speech_0.mp3
│   ├── speech_1.mp3
│   └── ...
└── README.md
```

## Functionality Overview

1. **Initialization**

   - Imports necessary libraries and modules.
   - Checks for GPU availability and sets the device accordingly.
   - Initializes the OCR reader and text-to-speech engine.
   - Loads the YOLO models for object detection.
   - Sets up video capture and writer objects.
   - Pre-generates speech clips for common phrases to optimize performance.

2. **Processing Loop**

   - Reads frames from the video file.
   - Skips frames based on a defined interval to reduce processing time.
   - Performs object detection on each frame using the YOLO model.
   - Identifies specific classes and processes them:
     - **Speed Limit Signs**: Uses EasyOCR to read and display the speed limit.
     - **Arrows**: Placeholder for direction detection.
     - **Traffic Lights**: Uses a secondary model to detect the color of the light.
     - **Other Classes**: Displays the class name.
   - Generates speech alerts when a new object class is detected.
   - Writes the processed frame to the output video.

3. **Post-processing**

   - Releases video capture and writer resources.
   - Combines the processed video with the generated audio clips to produce the final output.

## Customization

- **Frame Skipping**

  Adjust `frame_skip_interval` to process frames at different intervals.

  ```python
  frame_skip_interval = 1  # Process every frame
  ```

- **Model Selection**

  Replace the model paths with your custom models if needed.

  ```python
  model = YOLO('path_to_your_custom_model.pt')
  ```

- **Classes and Labels**

  Modify the code to handle additional classes or change labels as per your requirements.

- **Text-to-Speech Settings**

  Adjust the speech rate, volume, or voice properties in the `pyttsx3` engine initialization.

  ```python
  engine = pyttsx3.init()
  engine.setProperty('rate', 150)
  engine.setProperty('volume', 0.9)
  ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your_feature`.
3. Commit your changes: `git commit -am 'Add your feature'`.
4. Push to the branch: `git push origin feature/your_feature`.
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Disclaimer**: Ensure you have the rights and permissions to use any models and libraries included in this project. This code is provided as-is for educational purposes, and the author is not responsible for any misuse.
