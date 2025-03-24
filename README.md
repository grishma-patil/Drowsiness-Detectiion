## ***Drowsiness Detection***
A deep learning-based drowsiness detection system using OpenCV, TensorFlow, and dlib to monitor driver alertness in real-time.

## **Project Overview**

*This project aims to prevent road accidents by detecting driver drowsiness using real-time eye movement and facial landmarks. The system continuously monitors the eye aspect ratio (EAR) and triggers an alert when the driver shows signs of drowsiness.

## **Features**

*✔ Real-time eye tracking using OpenCV & dlib
*✔ Deep learning-based CNN model for high accuracy
*✔ Facial landmark detection to analyze eye movements
*✔ Alarm system for drowsiness alerts

## **Dataset & Model**

*Facial landmark detection is done using dlib's pre-trained model.
*CNN model is built using TensorFlow to classify drowsy and awake states.

## **How It Works?**

*1️⃣ Capture a video stream using OpenCV.
*2️⃣ Detect facial landmarks & extract eye coordinates.
*3️⃣ Calculate Eye Aspect Ratio (EAR) to monitor blink rate.
*4️⃣ Classify if the driver is drowsy or alert using the CNN model.
*5️⃣ Trigger an Alert if drowsiness is detected.

## **Model Performance**

*Achieved high accuracy in detecting drowsiness.
*Optimized with real-time processing for quick response.
*Tested on various lighting conditions and facial structures.

## **File Structure**
*Drowsiness_Detection_code.py - Main script for real-time detection
*Dataset - Contains images used for training
*README.md - Project documentation
