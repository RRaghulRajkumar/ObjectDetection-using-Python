import cv2
from random import randrange
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #classifier-detector

#to capture video
webcam =cv2.VideoCapture(0)

#iterate forever for frames
while True:
    #Read the current frames
    successful_frame_read,frame=webcam.read()
#must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #to detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)#detectMultiScale will datect objects of different sizes in the input image.
#detected objects are returned as rectangles
# [[220 162 162 162]]-output-where the face is present

#draw rectangles around the faces
    for(x,y,w,h) in face_coordinates:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),4)#128 to 256 bright colours
    #cv2.rectangle(img,((220,162),(162,162)),greencolor(b,g,r),thickness of rectangle)

    #to display image
    cv2.imshow('_face_detection',frame)

#waitkey is to pause the image      # 0 indicates infinity secondsfa
    cv2.waitKey(1)

#stop if Q key is pressed    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Release the videocapture object
 
cv2.destroyAllWindows()