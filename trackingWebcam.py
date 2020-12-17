import cv2
import sys

faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCasc.detectMultiScale(grayscale, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("img", img)
    ky = cv2.waitKey(30) & 0xff
    if ky == 27:
        break
cap.release()
cv2.destroyAllWindows()