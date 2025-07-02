# AI Fitness Trainer with Face Recognition & Pose Estimation

This is a Python-based **AI Fitness Trainer** that uses your webcam to:

- Recognize your face using pre-captured images
- Detect your body pose in real time using MediaPipe
- Count repetitions of exercises like **arm curls** using joint angle calculations

Built with:

- OpenCV for webcam and image processing
- MediaPipe for pose estimation
- face_recognition for face matching
- NumPy for math operations
- Threading for non-blocking processing

---

## Features

- Real-time face recognition for personalized workout tracking
- Accurate pose detection and angle-based rep counting
- Auto-captures and saves user images on first use
- Repetition counter appears on-screen live
- Keeps each user's face data in separate folders

---

## How to Download This Project

1. Go to the GitHub repository
2. Click the green **"Code"** button and choose **"Download ZIP"**
3. Extract the ZIP file on your system

---

## 🧪 Setup Instructions

1.Prerequisites
   - Python 3.9 (more suitable) or higher
   - Webcam
   - Windows/macOS/Linux system

2.Open Terminal in the Project Folder

3.Create and Activate Virtual Environment
     python -m venv .venv
     .venv\Scripts\activate

4.Install Required Packages
     pip install opencv-python face_recognition mediapipe numpy

5.Run the Project
     python Main.py

6.Prepare Face Database
     1. Run the project
     2. Enter your name when prompted
     3. The script will:
         - Create a folder for your images
         - Guide you through capturing training images
         - Store these for future recognition

7.Usage Instructions

     1. The system will:
         - Recognize your face (if trained)
         - Track your left arm for bicep curls
         - Count reps when your elbow angle goes below 36°
     2. Controls:
         - Spacebar: Toggle continuous image capture (during training)
         - 'q': Quit the application
