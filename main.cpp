import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = 'img'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curr_Img = cv2.imread(f'{path}/{cls}')
    images.append(curr_Img)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    print(encodeList)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv','r+') as f:# r+ for read and wrtie both operation
        myDataList = csv.reader(f)
        
        nameList = []
        for line in myDataList:
            print(myDataList)
            nameList.append(myDataList)

        print(name+"def Mark attendence")
        print(nameList)

        if name not in nameList:
            now = datetime.now()
            dtStr = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtStr}')

    

encodeListKnown = findEncodings(images)
print('Encoding complete')


# web Cams
cap = cv2.VideoCapture(0) # we are giving 0 as our id

while True : 
    success , img = cap.read()
    # to reduce the size of Image as we are capturing image in real time
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # here we have scale down the image 1/4th time.
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    # now in webcam there might be many faces in a webcam , we need find our face location.
    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurrFrame)

    # now we will iterate through all the face which we have found in our current frame  and compare all these face with the encodings we have found before.
    for encodeFace , faceLoc in zip(encodeCurrFrame,faceCurrFrame): # we want both parameter in same look so we are using zip
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis) # return index number of minimum element

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name+"if condition")
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 # here we are scaling up the calculated value from the scaled down image which was of 1/4th of the actual image.
            # so we can use our normal calcuation in scaled down image.
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-32),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img , name , (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            markAttendence(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
