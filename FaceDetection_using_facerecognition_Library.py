import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt

image = cv2.imread('Input Image Path')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_locations = []
rgb_frame = image[:, :, ::-1]

#Find all the faces in the current frame of video
face_locations = face_recognition.face_locations(rgb_frame)

for top, right, bottom, left in face_locations:
    #Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)

    roi_color = image[top:bottom,left:right]
    cv2.imwrite('Output Image Path', roi_color)
