import numpy as np
import cv2
import tkinter as tk
import PIL
import PIL.Image, PIL.ImageTk
from tkinter.ttk import *
from tkinter import *
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter 

loop = False
loop2 = False
class Runner(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)
        self.frames = {}

        frame = StartPage(container, self)

        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
class StartPage(tk.Frame):
 
    def __init__(self, window, controller, video_source=0): # https://192.168.43.1:8080/video
        tk.Frame.__init__(self,window)
        self.video_source = video_source
        self.controller = controller
        cam_frame = tk.Frame(self)
        self.vid = MyVideoCapture(self.video_source)
        self.stop_camera()
        self.canvas = tk.Canvas(cam_frame, width = self.vid.width-3, height = self.vid.height-3,bg="black",bd=2)
        self.delay = 15
        self.canvas.pack(padx=5,pady=5)
        cam_frame.pack(side="left",anchor="nw")
        self.center_coordinates = (int(self.vid.width//2),int(self.vid.height//2)) 
        self.radius = int(self.vid.height//3)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.Blue = (0, 0, 255)
        self.Red =  (255,0,0)
        self.Green = (0,255,0)
        self.Black = (0,0,0)
        self.thickness = 2
        self.org = ((self.vid.width//2)-100, self.vid.height-25) 
        self.fontScale = 1
        self.maskNet = load_model("Weights\mask_type.h5")
        self.eye_cascade = cv2.CascadeClassifier('Weights\haarcascade_eye.xml')
        prototxtPath = os.path.sep.join(["Weights", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["Weights","res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.counter = 0 
        self.mask_lst = []
        self.forhead_lst = []
        self.start_camera()
        
    def detect_and_predict_mask(self,frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > .2:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.resize(face, (224, 224))
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
        return (len(faces),locs)
    
    def add_display_info(self,Numfaces,locs,frame2,w,h):
        if Numfaces >= 1:
            (startX, startY, endX, endY) = locs[0]
            if abs(endX-startX) > ((w//2)) and abs(endY-startY) > ((h//2)):
                cc = self.Green
                lst = ["Cotton_Mask","N-95_Mask","No_Mask","Three_Layer_Mask"]
                img = cv2.resize(frame2,(224,224))/255.0
                pred = self.maskNet.predict(img.reshape(1,224,224,3))
                label = lst[np.argmax(pred)]
                self.mask_lst.append(label)
                coor = (self.center_coordinates[0]-(len(label)*8),self.frame.shape[0]-10)
                self.frame = cv2.putText(self.frame,"Correct",(self.center_coordinates[0]-50,self.center_coordinates[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
            else:
                cc = self.Red
                self.frame = cv2.putText(self.frame,"To Far From Screen",(self.center_coordinates[0]-158,self.center_coordinates[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        else:
            cc = self.Black
        self.frame = cv2.circle(self.frame, self.center_coordinates, self.radius, cc, self.thickness)

    def detect_eyes(self,frame):
        eyes = self.eye_cascade.detectMultiScale(frame)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(self.frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    def update(self):
        try:
            global loop
            if loop:
                x,y,w,h = self.center_coordinates[0]-self.radius,self.center_coordinates[1]-self.radius,2*self.radius,2*self.radius
                self.ret, self.frame = self.vid.get_frame()
                frame2 = self.frame[y:y+h,x:x+w,:].copy()
                Numfaces,locs = self.detect_and_predict_mask(frame2)
                if len(self.mask_lst) != 40:
                    self.add_display_info(Numfaces,locs,frame2,w,h)
                else:
                    self.detect_eyes(self.frame)
                if len(self.mask_lst) >1:
                    ll = Counter(self.mask_lst).most_common(1)[0][0]
                    self.frame = cv2.putText(self.frame,str(ll),(x+20,y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
                if self.ret:
                    self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame))
                    self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
                    self.after(self.delay, self.update)
        except Exception as e:
            print(e)
            print("Restart Application")
            self.stop_camera()
    
    def stop_camera(self):
        global loop
        if loop:
            loop = False
            self.vid.stop_video_capture()
            self.canvas.delete("all")
            
    def start_camera(self):
        global loop
        if not loop:
            loop = True
            self.vid.start_video_capture()
            self.update()
class MyVideoCapture:
    
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.vid1 = cv2.VideoCapture(self.video_source)
        if not self.vid1.isOpened():
             raise ValueError("Unable to open video source", video_source)
        self.width = self.vid1.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def start_video_capture(self):
        if not self.vid1.isOpened():
            print("start_video_capture")
            self.vid1 = cv2.VideoCapture(self.video_source)
      
    def stop_video_capture(self):
        if self.vid1.isOpened():
            print("stop_video_capture")
            self.vid1.release()
                
    def get_frame(self):  
        if self.vid1.isOpened():
            ret, frame = self.vid1.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

app = Runner()
app.mainloop()





