# 6. Inference for real-time webcam emotion detection (run after training or after loading model)
from tensorflow.keras.models import load_model
import cv2
import numpy as np
model = load_model('best_emotion_vgg16.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi_rgb = cv2.resize(roi_rgb, (224,224))
        roi = roi_rgb.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(pred)]
        confidence = np.max(pred)
        label_text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
