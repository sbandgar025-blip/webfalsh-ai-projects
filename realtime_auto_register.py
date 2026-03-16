import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

DB_FILE = "face_db.npz"

# Thresholds (you can tune these later)
RECOGNITION_THRESHOLD = 0.50

# Cosine similarity
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Load database
if not os.path.exists(DB_FILE):
    print("[ERROR] face_db.npz not found! Run register_faces.py first.")
    exit()

db = np.load(DB_FILE, allow_pickle=True)
known_embeddings = db["embeddings"]
known_names = db["names"]

print("[INFO] Loaded database with", len(known_names), "faces")

# Load model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



print("[INFO] Starting camera... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        emb = face.embedding

        best_score = -1
        best_name = "Unknown"

        # Compare with known faces
        for i in range(len(known_embeddings)):
            score = cosine_sim(emb, known_embeddings[i])
            if score > best_score:
                best_score = score
                best_name = known_names[i]

        # Threshold decision
        if best_score < RECOGNITION_THRESHOLD:
            best_name = "Unknown"

        # Draw box and name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_name} ({best_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()
