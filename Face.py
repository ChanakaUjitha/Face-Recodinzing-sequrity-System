import numpy as np
import cv2
import pickle
from email.message import EmailMessage
import mimetypes
import smtplib
import time
import ssl

#mail send class
def sendmail():
    Email_Address = 'chanuslash@gmail.com'
    Email_Password ='chanu911030160'
    context = ssl.create_default_context()
    msg = EmailMessage()
    msg['Subject'] = 'Unotherized Person Alert!'
    msg['from'] = Email_Address
    msg['To'] = 'chanakaujitha@gmail.com'
    msg.set_content('image attachment.......!')

    mime_type, _ = mimetypes.guess_type('save.jpg')
    mime_type, mime_subtype = mime_type.split('/')

    with open('save.jpg', 'rb') as file:
        msg.add_attachment(file.read(),
        maintype=mime_type,
        subtype=mime_subtype,
        filename='save.jpg')
    #print(msg)


    #
    mail_server = smtplib.SMTP('smtp.gmail.com',587)
    #mail_server.set_debuglevel(1)
    mail_server.starttls(context=context)
    mail_server.login(Email_Address,Email_Password)
    mail_server.send_message(msg)
    mail_server.quit()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data/train/trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb')as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y + h, x:x + w]
        

        id_, conf = recognizer.predict(roi_gray)
        conf = 100 - int(conf)
        pred = 0
        shot = 0
        if conf >= 45:  # and conf<=85:
            pred += +1
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(frame, name, (x, y-4), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        else:  
            pred += -1
            text = "Unkown Person"
            font = cv2.FONT_HERSHEY_PLAIN
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)
            shot = 32

            #unknow person photo save local
            
            _,img = cap.read()
            cv2.imshow('img' , img)
            
            print("image saved")
            file = 'save.jpg'
            cv2.imwrite(file,img)

            #sendmail()
            print("mail send")
            


           
       
        

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






