import cv2
import sys
import pickle
import numpy as np

image = sys.argv[1]
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}
with open ("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

img = cv2.imread(image)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4, minSize=(30, 30))


for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x: x + w]
    roi_color = img[y:y + h, x: x+ w]
    id_, conf = recognizer.predict()
    if conf >= 45 and conf <= 85:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(img, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Faces", img)
    cv2.waitKey(0)

