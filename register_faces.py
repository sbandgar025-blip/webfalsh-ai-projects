import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

KNOWN_DIR = "known_faces"
OUTPUT_FILE = "face_db.npz"

# Load InsightFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 = CPU

embeddings = []
names = []

print("[INFO] Registering known faces...")

for person in os.listdir(KNOWN_DIR):
    person_path = os.path.join(KNOWN_DIR, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"[SKIP] No face found in {img_path}")
            continue

        # take the biggest face
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )

        emb = faces[0].embedding

        embeddings.append(emb)
        names.append(person)

        print(f"[OK] Added {person} -> {img_name}")

embeddings = np.array(embeddings)
names = np.array(names)

np.savez(OUTPUT_FILE, embeddings=embeddings, names=names)

print(f"[DONE] Saved {len(names)} faces into {OUTPUT_FILE}")
print("[DONE] Database file created:", OUTPUT_FILE)
