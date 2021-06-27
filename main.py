# import the opencv library
import cv2
import imutils
import numpy as np
#face_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
car_cascade = cv2.CascadeClassifier('cars.xml')
# define a video capture object
vid = cv2.VideoCapture("videos_cars.mp4")
rojo = (0, 0, 255)
azul = (255, 0, 0)
verde = (0, 255, 0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def rgb_to_temperature(x, y, w, h, color):
    max = 255
    for i in np.nditer(color):
        if (i) < max:
            max = i
    print(max)
    temperatura = 255/100.0 * (max)
    return temperatura

while (True):
    ret, frame = vid.read()
    #Se voltea la imagen para que sea natural
    frame = cv2.flip(frame, 1)

    #Se redimenciona la imagen para que se haga mas liviano el
    #analisis de los pixeles
    frame = imutils.resize(frame,width=min(400, frame.shape[1]))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    (regions, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05)
    
    #Transformacion de video a mapa de calor
    calor = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

    #Se dibura el buldingbox en los objetos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), azul, 2)
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rojo, 2)
    for (x, y, w, h) in cars:
        cv2.putText(calor, 'La T es: '+ str(rgb_to_temperature(x,y, w,h, gray)), (x- 10, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), verde, 2)
    cv2.imshow('frame', frame)
    
    visual = np.concatenate((frame, calor), axis=1)
    cv2.imshow('HORIZONTAL', visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
