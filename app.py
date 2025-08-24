from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import re
from flask import send_from_directory
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Load model once on startup
model = load_model('best_emotion_vgg16.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    normalized = resized.astype('float32') / 255.0
    expanded = np.expand_dims(normalized, axis=0)
    return expanded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']  # Expect base64 string with data:image/jpeg;base64,...

    # Decode base64 image
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    (x, y, w, h) = faces[0]  # Using first detected face
    roi_gray = gray[y:y+h, x:x+w]
    roi_input = preprocess_face(roi_gray)

    prediction = model.predict(roi_input)[0]
    emotion = emotion_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({'emotion': emotion, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
