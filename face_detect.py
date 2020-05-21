# -*- coding: utf-8 -*-
"""
Created on Thu May 21 03:02:47 2020

@author: Bqasx
"""

import cv2, numpy as np, dlib, pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from gtts import gTTS
import playsound
import os


scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("python").sheet1  # Open the spreadhseet


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC,FACE_NAME = pickle.load(open('trainset.pk','rb'))
cap = cv2.VideoCapture(1)

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        dets = detector(img,1)
        for k,d in enumerate(dets):
            shape = sp(img, d)
            face_desc0 = model.compute_face_descriptor(img,shape,1)
            
            
            d=[]
            for face_desc in FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
            d = np.array(d)
            idx = np.argmin(d)
            if d[idx] < 0.5:
                name = FACE_NAME[idx]
                
                
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, .7 , (255,255,255),2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                
                x = datetime.datetime.now()
                time=x.strftime("%H:%M:%S")
                date =x.strftime("%d/%m/%Y") 
                col = sheet.col_values(1)
                date_col = sheet.col_values(3)
                if name not in col :
                    
                    tts=gTTS(text='สวัสดีค่ะคุณ'+name,lang='th')
                    tts.save('sound_out.mp3')
                    playsound.playsound('sound_out.mp3',True)
                    os.remove("sound_out.mp3")
                    
                    time=x.strftime("%H:%M:%S")
                    date =x.strftime("%d/%m/%Y") 
                    insert_row = [name,time,date]
                    sheet.insert_row(insert_row,len(col)+1)
                    
                    
             
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
    