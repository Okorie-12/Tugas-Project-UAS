# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 06:44:37 2024

@author: ASUS
"""

import cv2
import os

# karena saya nggk bisa menaruh satu direktori pada open cv jadi saya direk manual ke folder kodingan
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier("C:\\Users\ASUS\Downloads\presentasi\\haarcascade_frontalface_default.xml")

# memulai video capture (nilai 0 ada settingan default videocapture)
cap = cv2.VideoCapture(0)

# Check jika videocapture terbuka secara proper
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # konversi frame menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # mengambar bentuk kotak pada area wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('Face Detection', frame)

    # jika pengen close run program dengan tombol
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
