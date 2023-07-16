import cv2
import numpy as np
import face_recognition

# importing images
image1 = face_recognition.load_image_file('Training_images/ElonMusk.jpeg')
# convert to rgb
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Training_images/Sudhanshu.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#  finding faces in our image and encoding it
faceLoc=face_recognition.face_locations(image1)[0]
#  sending first element of it
encodeElon= face_recognition.face_encodings(image1)[0]
#  seeing the face locations
cv2.rectangle(image1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# test image
faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest= face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# comparing faces
results = face_recognition.compare_faces([encodeElon],encodeTest)

# printing distances among the 2 images
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Elon Musk' , image1)
cv2.imshow('Elon Test' , imgTest)

cv2.waitKey(0)


