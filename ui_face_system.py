import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from insightface.app import FaceAnalysis

DB_FILE = "face_db.npz"

try:
    db = np.load(DB_FILE, allow_pickle=True)
    known_embeddings = list(db["embeddings"])
    known_names = list(db["names"])
except:
    known_embeddings = []
    known_names = []

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.withdraw()

print("Press R to register unknown person")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        emb = face.embedding
        x1,y1,x2,y2 = map(int,face.bbox)

        best_score = 0
        name = "Unknown"

        for i,db_emb in enumerate(known_embeddings):
            score = cosine_sim(emb,db_emb)

            if score > best_score:
                best_score = score
                name = known_names[i]

        if best_score < 0.5:
            name = "Unknown"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Face System",frame)

    key = cv2.waitKey(1)

    if key == ord('r'):

        if len(faces) > 0:

            new_name = simpledialog.askstring("Register","Enter Name")

            if new_name:

                known_embeddings.append(faces[0].embedding)
                known_names.append(new_name)

                np.savez(DB_FILE,
                         embeddings=known_embeddings,
                         names=known_names)

                print("Saved:",new_name)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()