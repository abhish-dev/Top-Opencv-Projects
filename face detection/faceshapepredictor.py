import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the pre-trained age transformation model
model_path = r'C:\Users\Abhishek Kumar\OneDrive\Videos\Documents\GitHub\collaborative project\Top-Opencv-Projects\face detection\age\model.h5'
try:
    age_model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize MediaPipe Face Detection and Drawing
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Initialize video capture
cap = cv2.VideoCapture(0)

def preprocess_face(face_image):
    # Resize and normalize the face image to match the model input
    face_image = cv2.resize(face_image, (128, 128))  # Adjust size as needed
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def transform_face(face_image):
    # Apply the model to the face image
    preprocessed_image = preprocess_face(face_image)
    # Simulate age transformation (assuming model predicts age)
    predicted_age = age_model.predict(preprocessed_image)
    # Transform face based on predicted age (Placeholder - you need a specific transformation)
    transformed_face = face_image  # Replace with actual transformation logic
    return transformed_face

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = frame[y:y+h, x:x+w]

            # Apply face transformation
            transformed_face = transform_face(face_image)

            # Replace face in the original image with the transformed face (Optional)
            frame[y:y+h, x:x+w] = transformed_face

            # Draw bounding box
            mp_drawing.draw_detection(frame, detection)

    # Display the result
    cv2.imshow('Face Transformation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
