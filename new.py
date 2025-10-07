import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential  # core class

model = load_model('your_model.h5', custom_objects={'Sequential': Sequential})
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="🎭",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-result {
        font-size: 2.5rem;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_emotion_model():
    with open('emotion_model3.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("emotion_model3.weights.h5",custom_objects={'Sequential': Sequential})

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return model, face_cascade

# ---------------- EMOTION DATA ----------------
emotion_dict = {
    0: "😠 Angry",
    1: "🤢 Disgusted",
    2: "😨 Fearful",
    3: "😊 Happy",
    4: "😐 Neutral",
    5: "😢 Sad",
    6: "😲 Surprised"
}

emotion_colors = {
    0: "#FF6B6B",  # Angry
    1: "#95E1D3",  # Disgusted
    2: "#A8E6CF",  # Fearful
    3: "#FFD93D",  # Happy
    4: "#C7CEEA",  # Neutral
    5: "#6C5CE7",  # Sad
    6: "#FFA07A"   # Surprised
}

# ---------------- DETECTION FUNCTION ----------------
def detect_emotion(image, model, face_cascade):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None, img, 0

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(cropped_img, axis=-1)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = cropped_img / 255.0

        prediction = model.predict(cropped_img, verbose=0)
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]
        confidence = float(np.max(prediction)) * 100

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        results.append({'emotion': emotion, 'confidence': confidence, 'index': max_index})

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return results, img, len(faces)

# ---------------- MAIN HEADER ----------------
st.markdown('<h1 class="main-header">🎭 Face Emotion Detection</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem; color: #666;'>
Upload an image or use your webcam to detect emotions in real time!
</p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    with st.spinner("Loading AI model..."):
        emotion_model, face_cascade = load_emotion_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ---------------- MODE SELECTION ----------------
mode = st.radio("Choose Input Method", ["🖼️ Upload Image", "📷 Use Webcam"], horizontal=True)

# ---------------- FILE UPLOAD MODE ----------------
if mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'],
                                     help="Upload a clear photo with visible faces")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Detect Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing emotions..."):
                results, processed_img, num_faces = detect_emotion(image, emotion_model, face_cascade)

            if results is None:
                st.error("❌ No faces detected. Please upload a clearer photo.")
            else:
                with col2:
                    st.subheader("🎯 Detection Result")
                    st.image(processed_img, use_container_width=True)

                st.markdown("---")
                st.subheader(f"📊 Detected {num_faces} Face(s)")

                for i, result in enumerate(results, 1):
                    emotion = result['emotion']
                    confidence = result['confidence']
                    color = emotion_colors[result['index']]

                    st.markdown(f"""
                        <div style='background-color: {color}; padding: 1rem; 
                        border-radius: 10px; margin: 1rem 0;'>
                            <h2 style='margin: 0; text-align: center;'>{emotion}</h2>
                            <p style='text-align: center; font-size: 1.2rem; margin: 0.5rem 0;'>
                                Confidence: {confidence:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(confidence / 100)

# ---------------- WEBCAM MODE ----------------
elif mode == "📷 Use Webcam":
    st.info("Click below to capture your face from the webcam.")
    camera_input = st.camera_input("Capture Image")

    if camera_input is not None:
        image = Image.open(camera_input)

        with st.spinner("Analyzing live image..."):
            results, processed_img, num_faces = detect_emotion(image, emotion_model, face_cascade)

        if results is None:
            st.warning("😕 No face detected. Try again with better lighting.")
        else:
            st.subheader("🎯 Detection Result")
            st.image(processed_img, use_container_width=True)

            st.markdown("---")
            st.subheader(f"📊 Detected {num_faces} Face(s)")

            for i, result in enumerate(results, 1):
                emotion = result['emotion']
                confidence = result['confidence']
                color = emotion_colors[result['index']]

                st.markdown(f"""
                    <div style='background-color: {color}; padding: 1rem; 
                    border-radius: 10px; margin: 1rem 0;'>
                        <h2 style='margin: 0; text-align: center;'>{emotion}</h2>
                        <p style='text-align: center; font-size: 1.2rem; margin: 0.5rem 0;'>
                            Confidence: {confidence:.1f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(confidence / 100)

# ---------------- SIDEBAR INFO ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI model can detect 7 emotions:
    - 😠 Angry  
    - 🤢 Disgusted  
    - 😨 Fearful  
    - 😊 Happy  
    - 😐 Neutral  
    - 😢 Sad  
    - 😲 Surprised
    """)

    st.header("🛠️ Tech Stack")
    st.write("""
    - TensorFlow/Keras  
    - OpenCV  
    - Streamlit  
    - Python
    """)

    st.header("💡 Tips")
    st.write("""
    - Use clear, well-lit photos  
    - Face should be frontal  
    - Works with multiple faces  
    - Webcam supports live detection
    """)

    st.markdown("---")
    st.markdown("Made with ❤️ by **Sanika Surashe**")

