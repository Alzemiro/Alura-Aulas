import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('jovem.jpg',0)

image_grey = cv2.cvtColor(image,cv2.COLOR_BAYER_BG2GRAY)

faces = face_classifier.detectMultiScale(image_grey, 1.3, 5)

for (x,y,h,w) in faces:
    cv2.rectangle(image,(x,y), (x+w,y+h),(255,255,0),2)

cv2.imshow('imagem', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


