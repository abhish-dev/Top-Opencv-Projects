import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to draw face mesh on the image
def draw_face_mesh(image, face_landmarks):
    for landmarks in face_landmarks:
        for i in range(0, len(landmarks.landmark)):
            x = int(landmarks.landmark[i].x * image.shape[1])
            y = int(landmarks.landmark[i].y * image.shape[0])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
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
        frame = draw_face_mesh(frame, results.multi_face_landmarks)

    # Display the frame
    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
