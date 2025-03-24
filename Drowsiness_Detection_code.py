import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
import imutils

# Load pre-trained deep learning model for eye state classification
model = load_model("drowsiness_model.h5")  # Ensure you have a trained CNN model

# Load dlib’s face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)  # Compute EAR
    return ear

# Define EAR threshold for drowsiness
EAR_THRESHOLD = 0.25  
FRAME_CONSECUTIVE = 20  # Consecutive frames threshold
frame_counter = 0

# Start Video Capture
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract Eye Regions
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Use Deep Learning Model for Eye Classification
        eye_frame = gray[left_eye[1][1]:left_eye[4][1], left_eye[0][0]:left_eye[3][0]]
        eye_frame = cv2.resize(eye_frame, (24, 24)) / 255.0
        eye_frame = np.expand_dims(eye_frame, axis=0)
        eye_frame = np.expand_dims(eye_frame, axis=-1)
        prediction = model.predict(eye_frame)
        eye_state = "Open" if prediction[0][0] > 0.5 else "Closed"

        # Draw eye landmarks
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # If EAR is below threshold for consecutive frames, raise alert
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= FRAME_CONSECUTIVE:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            frame_counter = 0

        # Display EAR and Eye State
        cv2.putText(frame, f"EAR: {avg_EAR:.2f} | Eye: {eye_state}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show Frame
    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
