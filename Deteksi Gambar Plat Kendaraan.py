# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 06:52:24 2024

@author: ASUS
"""

import cv2
import pytesseract

# Specify path to the Tesseract executable (if not in PATH)
# For example, on Windows, it could be something like:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image containing the license plate
image_path = "C:\\Users\ASUS\Downloads\presentasi\\Plat Kendaraan.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply some preprocessing to improve OCR accuracy
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise
edges = cv2.Canny(gray, 170, 200)             # Detect edges

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Get top 10 contours

license_plate = None

# Loop through contours to find a potential license plate
for contour in contours:
    # Approximate the contour to see if it has a rectangular shape
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

    # If the contour has 4 vertices, it may be a license plate
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        license_plate = gray[y:y+h, x:x+w]
        break

if license_plate is None:
    print("License plate not found")
else:
    # Optional: Apply further thresholding for better OCR accuracy
    _, license_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract OCR to read text from the license plate
    text = pytesseract.image_to_string(license_plate, config='--psm 8')
    print("Detected License Plate Text:", text)

    # Display the detected license plate
    cv2.imshow("License Plate", license_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
