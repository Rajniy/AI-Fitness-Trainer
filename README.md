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

## 🎯 Features

- Real-time face recognition for personalized workout tracking
- Accurate pose detection and angle-based rep counting
- Auto-captures and saves user images on first use
- Repetition counter appears on-screen live
- Keeps each user's face data in separate folders

---

## ⬇️ How to Download This Project

1. Go to the GitHub repository
2. Click the green **"Code"** button and choose **"Download ZIP"**
3. Extract the ZIP file on your system

---

## 🛠 Setup Instructions

### Prerequisites
   - Python **3.9** or higher  
   - A working webcam  
   - Works on **Windows**, **macOS**, and **Linux**

### Installation & Setup
1. **Open Terminal** in the project folder  

2. **Create and activate a virtual environment**  

       python -m venv .venv
       .venv\Scripts\activate    # For Windows
       source .venv/bin/activate # For macOS/Linux

3.Install Required Packages
   
     pip install opencv-python face_recognition mediapipe numpy

4.Run the Project
   
     python Main.py

5.📷 Preparing Face Database
When you run the project for the first time:
1.You'll be prompted to enter your name
2.The system will:
- Create a folder for your images
- Guide you through capturing training images
- Store these for future recognition

6.Usage Instructions
  
1. The system will:
- Recognize your face (if trained)
- Track your left arm for bicep curls
- Count a rep when your elbow angle drops below 36°
2. Controls:
- Spacebar: Toggle continuous image capture (during training)
- 'q': Quit the application
