import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/home/vishal/opencv/data/lbpcascades/lbpcascade_frontalface.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.createLBPHFaceRecognizer();
rec.load('/home/vishal/BTP_Final/trainingdata.yml')
lebel=0
#font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
#           
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    #locy = float(img.shape[0]/2) # the text location will be in the middle
    #locx = float(img.shape[1]/2) 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        lebel,conf=rec.predict(gray[y:y+h,x:x+w])
        if(lebel==2):
            lebel="HARRY"
        if (lebel==1):
            lebel="VISHAL"
        
        cv2.putText(img,str(lebel),(x,y+h),fontFace,fontScale, fontColor)
    cv2.imshow('img',img)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()

cv2.destroyAllWindows()


