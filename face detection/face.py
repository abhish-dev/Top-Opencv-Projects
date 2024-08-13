import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO

# Download image from URL
url = 'image jpg'
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

# Function to warp and overlay image with facial expression mapping
def overlay_image(src, dst, landmarks):
    if len(landmarks) < 6:
        print("Not enough landmarks detected.")
        return dst

    h, w, _ = dst.shape

    # Resize the target image to a smaller size
    target_size = (150, 150)
    resized_target = cv2.resize(src, target_size)

    # Define points for the target image (right side of the video feed)
    tx, ty = w - target_size[0] - 10, 10  # X and Y coordinates for positioning the image
    tx_end, ty_end = tx + target_size[0], ty + target_size[1]

    # Use facial landmarks for facial features
    # Ensure we have correct landmark indices
    src_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in [33, 61, 291, 199, 0, 17]]  # Example points
    dst_pts = [(tx, ty), (tx_end, ty), (tx_end, ty_end), (tx, ty_end), (tx + target_size[0]//2, ty + target_size[1]//2)]

    # Check if points are valid
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Not enough points for homography.")
        return dst

    src_pts = np.array(src_pts[:4], dtype=np.float32)
    dst_pts = np.array(dst_pts[:4], dtype=np.float32)

    try:
        # Compute homography and warp the source image
        M, _ = cv2.findHomography(src_pts, dst_pts)
        warped_image = cv2.warpPerspective(resized_target, M, (w, h))

        # Create a mask and blend the images
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(dst_pts, dtype=np.int32), 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = np.where(mask == 255, warped_image, dst)

        # Place the resized and warped image on the right side
        dst[ty:ty + target_size[1], tx:tx + target_size[0]] = resized_target

        return dst
    except cv2.error as e:
        print(f"Error in findHomography: {e}")
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
