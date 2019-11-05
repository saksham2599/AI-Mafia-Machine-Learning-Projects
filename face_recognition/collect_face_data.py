import cv2
import numpy as np

#init camera
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

face_data=[]
skip=0
path='./data/'
filename=input('enter your name Please!!!')
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f: f[2]*f[3])
    if len(faces)==0:
        continue
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #extract area of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
        
    cv2.imshow('Frame',frame)
    cv2.imshow('face_section',face_section)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)

np.save(path+filename+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()


