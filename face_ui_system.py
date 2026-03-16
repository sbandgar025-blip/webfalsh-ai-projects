import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
import csv
import datetime
import os

DB_FILE = "face_db.npz"

# Load database
try:
    db = np.load(DB_FILE, allow_pickle=True)
    known_embeddings = list(db["embeddings"])
    known_names = list(db["names"])
except:
    known_embeddings = []
    known_names = []

# Face model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Attendance logger
def mark_attendance(name):
    with open("attendance.csv","a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name,datetime.datetime.now()])

# Register new person
def register_face():
    global last_embedding
    name = simpledialog.askstring("Register","Enter Person Name")

    if name:
        known_embeddings.append(last_embedding)
        known_names.append(name)

        np.savez(DB_FILE,
                 embeddings=known_embeddings,
                 names=known_names)

        mark_attendance(name)
        print("Registered:",name)

# UI
root = tk.Tk()
root.title("AI Face Recognition System")

video_label = tk.Label(root)
video_label.pack()

btn = tk.Button(root,text="Register New Person",command=register_face)
btn.pack()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_embedding = None

def update_frame():
    global last_embedding

    ret,frame = cap.read()

    if ret:

        faces = app.get(frame)

        for face in faces:

            emb = face.embedding
            last_embedding = emb

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
            else:
                mark_attendance(name)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10,update_frame)

update_frame()

root.mainloop()

cap.release()