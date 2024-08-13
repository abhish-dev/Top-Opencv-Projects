import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO

# Download image from URL
url = 'https://i.pinimg.com/564x/49/86/4a/49864a1b8507154ba92f3f6d095b0e92.jpg'
response = requests.get(url)
img_data = BytesIO(response.content)
target_image = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), cv2.IMREAD_COLOR)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Function to get face landmarks
def get_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return []

# Function to draw landmarks on the image
def draw_landmarks(frame, landmarks):
    if landmarks:
        h, w, _ = frame.shape
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame

# Function to warp and overlay image
def overlay_image(src, dst, landmarks):
    if len(landmarks) < 4:
        return dst

    h, w, _ = dst.shape

    # Resize the target image to a smaller size
    target_size = (150, 150)
    resized_target = cv2.resize(src, target_size)

    # Define target image size and position (right side of the video feed)
    tx, ty = w - target_size[0] - 10, 10  # X and Y coordinates for positioning the image

    # Use specific facial landmarks (indices may vary, adjust as needed)
    src_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in [33, 61, 291, 199]]  # Example points
    dst_pts = [(tx, ty), (tx + target_size[0], ty), (tx + target_size[0], ty + target_size[1]), (tx, ty + target_size[1])]

    # Check if points are valid
    if len(src_pts) < 4:
        return dst

    # Compute homography and warp the source image
    M, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts))
    warped_image = cv2.warpPerspective(resized_target, M, (w, h))

    # Create a mask and blend the images
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(dst_pts, dtype=np.int32), 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = np.where(mask == 255, warped_image, dst)

    # Place the resized and warped image on the right side
    dst[ty:ty + target_size[1], tx:tx + target_size[0]] = resized_target

    return dst

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_landmarks(frame)
    frame_with_landmarks = draw_landmarks(frame, landmarks)
    frame_with_overlay = overlay_image(target_image, frame, landmarks)
    
    cv2.imshow('Webcam Feed', frame_with_overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
