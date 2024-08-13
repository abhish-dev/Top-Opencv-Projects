import cv2
import mediapipe as mp
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Function to download image from URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    return np.array(img)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the carnival mask image from URL
mask_url = 'image url'
mask_img = download_image(mask_url)
mask_img = cv2.resize(mask_img, (200, 200))  # Resize mask to a suitable size

# Function to overlay mask on the face
def overlay_mask(image, landmarks):
    for landmarks in landmarks:
        # Get bounding box for the face
        x_coords = [int(landmark.x * image.shape[1]) for landmark in landmarks.landmark]
        y_coords = [int(landmark.y * image.shape[0]) for landmark in landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Resize mask to fit the bounding box
        mask_resized = cv2.resize(mask_img, (x_max - x_min, y_max - y_min))
        
        # Overlay mask
        for i in range(mask_resized.shape[0]):
            for j in range(mask_resized.shape[1]):
                if mask_resized[i, j][3] != 0:  # Check alpha channel
                    image[y_min + i, x_min + j] = mask_resized[i, j][0:3]  # BGR values
    return image

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get face mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        frame = overlay_mask(frame, results.multi_face_landmarks)

    # Display the frame
    cv2.imshow('Face Mask Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
