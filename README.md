# Face-Emotion-Detection
An AI-powered web application that detects and classifies human emotions from facial expressions in real-time. Built with TensorFlow, OpenCV, and Flask/Streamlit.
Live Demo:
🎯 Problem Statement
Understanding human emotions is crucial in various domains like customer service, mental health monitoring, educational technology, and human-computer interaction. This project automates emotion recognition from facial expressions, making it accessible and scalable for real-world applications.
✨ Features

🎭 7 Emotion Classes: Detects Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised
👥 Multiple Face Detection: Can detect emotions for multiple faces in a single image
📊 Confidence Scores: Provides prediction confidence for transparency
🖼️ Visual Feedback: Highlights detected faces with bounding boxes and emotion labels
🌐 Web Interface: User-friendly interface for easy image upload and results
⚡ Real-time Processing: Fast inference for quick results

🛠️ Technology Stack
Machine Learning & Deep Learning

TensorFlow/Keras: Deep learning framework for model training
OpenCV: Computer vision library for face detection and image processing
NumPy: Numerical computing for array operations

Web Framework

Flask: Backend API server
Streamlit: Interactive web interface (alternative deployment)
Gradio: Quick ML demo interface (alternative deployment)

Face Detection

Haar Cascade Classifier: Pre-trained face detection algorithm
📊 Model Architecture
The emotion detection model is a Convolutional Neural Network (CNN) trained on facial expression datasets.
Key Specifications:

Input: 48x48 grayscale face images
Architecture: Multi-layer CNN with dropout for regularization
Output: 7-class softmax for emotion prediction
Training Dataset: FER-2013 
Accuracy: 70% on test set 
