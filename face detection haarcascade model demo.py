import cv2

alg = "C:/Users/jaine/Videos/OpenCV-face-detection-with-haarcascade_frontface_default.xml/haarcascade_frontalface_default.xml" #used haarcascade model
haar_cascade = cv2.CascadeClassifier(alg) #loading the model

cam = cv2.VideoCapture(0) #initializing the camera

while True:
    _,img = cam.read() #read the frame from the camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting color into gray

    face = haar_cascade.detectMultiScale(grayImg,1.3,4) #get coordinates of the face

    for (x,y,w,h) in face: #segregating x,y,w,h.
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0),2)
        cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
