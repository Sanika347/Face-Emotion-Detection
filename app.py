from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

from flask import Flask, render_template, request, jsonify
import os


import os
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'template'))


# Load model architecture
with open('emotion_model3.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)

# Load model weights
emotion_model.load_weights("emotion_model3.weights.h5")
print("✅ Loaded model from disk")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/Vishnu/Face-Emotion-Detection/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Failed to load Haar cascade file")

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/sanika')
def sanika():
    return render_template('sanika.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected'})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(cropped_img, axis=-1)
            cropped_img = np.expand_dims(cropped_img, axis=0)
            cropped_img = cropped_img / 255.0

            prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            emotion = emotion_dict.get(max_index, "Unknown")
            return jsonify({'emotion': emotion})

        return jsonify({'emotion': 'No face found'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
