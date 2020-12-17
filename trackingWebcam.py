import cv2
import sys
import numpy as np
import pickle

faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCasc.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x: x + w]
        roi_color = img[y:y + h, x: x+ w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("img", img)
    ky = cv2.waitKey(30) & 0xff
    if ky == 27:
        break
cap.release()
cv2.destroyAllWindows()