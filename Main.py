import cv2
import numpy as np
import face_recognition
import os
import mediapipe as mp
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to create a folder for the user's images
def create_folder(name):
    folder_path = f"{name}_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Function to check if a folder already exists
def folder_exists(name):
    folder_path = f"{name}_images"
    return os.path.exists(folder_path)

# Function to capture and save images
def capture_images(folder_path, num_images):
    cap = cv2.VideoCapture(0)
    images_captured = 0
    capture_continuous = False

    while images_captured < num_images:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Press 'Space' to Toggle Continuous Capture", frame)
            key = cv2.waitKey(1)
            if key == ord(' '):
                capture_continuous = not capture_continuous
            elif key == ord('q'):
                break
            elif capture_continuous or key == ord(' '):
                image_path = os.path.join(folder_path, f"image_{get_next_image_index(folder_path)}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image {images_captured + 1} saved as {image_path}")
                images_captured += 1
        else:
            print("Error: Cannot capture image.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to get the index for the next image
def get_next_image_index(folder_path):
    index = 0
    while os.path.exists(os.path.join(folder_path, f"image_{index}.jpg")):
        index += 1
    return index

# Function to find encodings of images
def findEncodings(images):
    encodelist = []
    for img in images:
        # Convert image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encode the face if it's detected
        face_encodings = face_recognition.face_encodings(imgRGB)
        if len(face_encodings) > 0:
            encode = face_encodings[0]
            encodelist.append(encode)
        else:
            print("No face detected in one or more images.")
    return encodelist

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize parameters for fine-tuning face encodings and implementing thresholds
tolerance = 0.6  # Tolerance for matching faces
confidence_threshold = 0.5  # Confidence threshold for recognizing faces

# Ask user for their name
user_name = input("Enter your name: ")

# Create a folder for the user's images if it doesn't exist
folder_path = create_folder(user_name)

# Check if the number of images in the folder exceeds 30
if len(os.listdir(folder_path)) >= 30:
    print("Number of images in the folder exceeds 30. Proceeding with face recognition directly.")
else:
    # Ask user for the total number of images to capture
    num_images = int(input("Enter the total number of images to capture: "))

    # Capture and save images
    capture_images(folder_path, num_images)

# Path to the directory containing folders of faces for recognition
path = f"{user_name}_images"

# Dictionary to store rep count for each recognized face
rep_counts = {}

# List to store images and class names (names of people)
images = []
classNames = []

# Check if the directory exists
if not os.path.exists(path):
    print(f"Error: Directory '{path}' does not exist.")
    exit()

# List files in the directory
files = os.listdir(path)

# Read images and extract class names
for file in files:
    img_path = os.path.join(path, file)
    curImg = cv2.imread(img_path)
    images.append(curImg)
    # Append folder name (class name) as the label
    classNames.append(user_name)

print("Class names:", classNames)

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Curl counter variables
stage = None

# Setup mediapipe instance for pose detection
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Recolor back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks for curl detection
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for the shoulder, elbow, and wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the elbow angle for curl counting
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle on the frame
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 36 and stage == 'down':
                stage = "up"
                # Check if a face is recognized
                if len(face_names) > 0:
                    recognized_face = face_names[0]
                    if recognized_face in rep_counts:
                        rep_counts[recognized_face] += 1
                    else:
                        rep_counts[recognized_face] = 1
                    print(f"{recognized_face} Reps: {rep_counts[recognized_face]}")

        except Exception as e:
            print(f"Pose detection error: {e}")

        # Render curl counter and face recognition status
        cv2.rectangle(image, (0, 0), (300, 73), (245, 117, 16), -1)

        # Display rep count for recognized faces
        text_y_position = 40
        for name, count in rep_counts.items():
            if name.upper() in rep_counts.keys():
                rep_text = f'{name.upper()} REPS: {count}'
                cv2.putText(image, rep_text, (15, text_y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                text_y_position += 30

        # Detect faces in the frame
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Compare the detected faces with known faces
        face_names = []
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=tolerance)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < confidence_threshold:
                # Get the name of the recognized person (folder name)
                name = classNames[matchIndex].upper()
                print("Recognized:", name)
                face_names.append(name)

                # Draw rectangle around the face and display the name
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Render pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Display the final output
        cv2.imshow('Combined Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    t1 = threading.Thread(target=findEncodings, name='t1', args=(images,))
    t1.start()
    t1.join()

    cap.release()
    cv2.destroyAllWindows()
