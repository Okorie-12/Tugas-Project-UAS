# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:14:21 2024

@author: ASUS
"""

import cv2

# Load the vehicle plate Haar cascade
plate_cascade = cv2.CascadeClassifier("C:\\Users\ASUS\Downloads\presentasi\\haarcascade_russian_plate_number.xml")

# Load the image where you want to detect the vehicle plate
image = cv2.imread("C:\\Users\ASUS\Downloads\presentasi\\rusia1.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect vehicle plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected plates
for (x, y, w, h) in plates:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected plates
cv2.imshow('Vehicle Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
