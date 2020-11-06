import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog,QMessageBox
from PyQt5 import uic
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QTimer

from centroidtracker import CentroidTracker
import collections
from copy import deepcopy

import time
import numpy as np
import cv2

Ui_MainWindow,QtBaseClass = uic.loadUiType("appUI.ui")


class main(QMainWindow):
    def __init__(self):
        super(main,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        '''
        First Start State
        '''
        self.ui.progressBar.hide()
        self.ui.runButton.setEnabled(False)
        self.ui.counterlineSpinbox.setEnabled(False)
        self.ui.objthreshSpinbox.setEnabled(False)
        self.ui.nmsthreshSpinbox.setEnabled(False)
        self.ui.maxdiasppearedSpinbox.setEnabled(False)
        self.ui.centroidthreshSpinbox.setEnabled(False)


        self.font = cv2.FONT_HERSHEY_PLAIN

        self.timer = QTimer()
        self.timer.setTimerType(0)


        '''
        all action
        '''
        
        self.ui.action_Exit.triggered.connect(self.exit_program)
        self.ui.action_About.triggered.connect(self.about)
        self.ui.action_loadVideo.triggered.connect(self.open_video)
        
        self.timer.timeout.connect(self.run_detector)
        self.ui.runButton.clicked.connect(self.controlTimer)
        self.ui.counterlineSpinbox.valueChanged.connect(self.draw_counterLine)
        self.ui.toplineSpinbox.valueChanged.connect(self.draw_counterLine)
        self.ui.bottomlineSpinbox.valueChanged.connect(self.draw_counterLine)

    def get_class(self,class_file):
        classes = []
        with open(class_file,"r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes   

    def detect_frame(self,frame,conf):
        height,width,channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > conf:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x-w/2)
                    y = int(center_y-h/2)
                    
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        return boxes,confidences,class_ids

    def controlTimer(self):
        if not self.timer.isActive():
            '''
            Prepare network
            '''
            weight = ["weight\\yolov3_custom_416.weights","weight\\yolov3_custom_512.weights"]
            cfg = ["cfg\\yolov3_custom_416.cfg","cfg\\yolov3_custom_512.cfg"]
            class_file = "coco.names"
            savename = "detect"

            weightIndex = self.ui.weightCombobox.currentIndex()
            print(weight[weightIndex],cfg[weightIndex])
            ##Load YOLO
            self.net = cv2.dnn.readNet(weight[weightIndex],cfg[weightIndex])
            #self.class_list = self.get_class(class_file)
            self.class_list = ["Pothole","Alligator Craci","Lateral Crack","Logitudinal Crack"]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
            self.colors = np.random.uniform(0,255,size=(len(self.class_list),3))
            


            self.frame = 0
            self.ui.progressBar.show()
            self.cap = cv2.VideoCapture(self.video_source)
            
            self.out_vid = cv2.VideoWriter(self.video_source.split("/")[-1].split(".")[0]+" ("+str(time.ctime(time.time())).replace(":",".")+")"+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'),self.framerate, (int(self.frameWidth),int(self.frameHeight)))
            
            ##tracker
            self.class_list_detect = [0,1,2,3]
            self.ct = []
            self.objects_temp = []
            for ct in range(len(self.class_list_detect)):
                self.ct.append(CentroidTracker(maxDisappeared=int(self.ui.maxdiasppearedSpinbox.value()),centroidThresh=int(self.ui.centroidthreshSpinbox.value())))


            ##Counter
            self.counter = [0 for i in range(len(self.class_list_detect))]
            self.counter_id_mem = [[] for i in range(len(self.class_list_detect))]


            self.timer.start(int(1000/self.framerate))
            self.start_time = time.time()
            self.ui.runButton.setEnabled(False)
            self.ui.counterlineSpinbox.setEnabled(False)
            self.ui.objthreshSpinbox.setEnabled(False)
            self.ui.nmsthreshSpinbox.setEnabled(False)
            self.ui.maxdiasppearedSpinbox.setEnabled(False)
            self.ui.centroidthreshSpinbox.setEnabled(False)
            

    def open_video(self):
        self.video_source = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Video File (*.mp4 *.avi)")[0]
        if self.video_source:

            cap = cv2.VideoCapture(str(self.video_source))
            self.frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
            self.framerate = cap.get(cv2.CAP_PROP_FPS)
            
            self.ui.source_label.setText(str(self.video_source))
            self.ui.dimension_label.setText('('+str(self.frameWidth)+'x'+str(self.frameHeight)+')  '+str(round(self.framerate,2))+' fps\n'+str(self.frameCount)+' frame')
            
            for i in range(1):
                ret,frame = cap.read()
                self.firstFrame = frame
                self.setMainScreen(frame)
                self.setSourceScreen(frame)

            cap.release()

            self.ui.counterlineSpinbox.setEnabled(True)
            self.ui.objthreshSpinbox.setEnabled(True)
            self.ui.nmsthreshSpinbox.setEnabled(True)
            self.ui.maxdiasppearedSpinbox.setEnabled(True)
            self.ui.centroidthreshSpinbox.setEnabled(True)
            self.ui.runButton.setEnabled(True)

        else:
            msg = QMessageBox()
            msg.setWindowTitle("Information")
            msg.setText("You haven't selected any video")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()

    def draw_counterLine(self):

        #getTopLine
        self.a1,self.b1 = 0,self.frameHeight*self.ui.toplineSpinbox.value()
        self.a2,self.b2 = self.frameWidth,self.frameHeight*self.ui.toplineSpinbox.value()

        #getButtonLine
        self.c1,self.d1 = 0,(self.frameHeight-self.b1)*self.ui.bottomlineSpinbox.value()+self.b1
        self.c2,self.d2 = self.frameWidth,(self.frameHeight-self.b2)*self.ui.bottomlineSpinbox.value()+self.b2

        #getCounterLine
        self.p1,self.q1 = 0,((self.d1-self.b1)*self.ui.counterlineSpinbox.value())+self.b1
        self.p2,self.q2 = self.frameWidth,((self.d2-self.b2)*self.ui.counterlineSpinbox.value())+self.b2
        
        
        
        frame = self.firstFrame.copy()
        frame = cv2.line(frame,(int(self.p1),int(self.q1)),(int(self.p2),int(self.q2)),(0,0,255),2)
        

        frame = cv2.line(frame,(int(self.a1),int(self.b1)),(int(self.a2),int(self.b2)),(0,211,255),2)
        frame = cv2.line(frame,(int(self.c1),int(self.d1)),(int(self.c2),int(self.d2)),(0,211,255),2)
        self.setMainScreen(frame)
        self.ui.runButton.setEnabled(True)

    def convert_detect(self,boxes,class_ids,indexes,class_list):
        '''
        return [[class_0],[class_1],...,[class_n]]
        with [class_n] = [x1,y1,x2,y2]
        '''
        detects = []
        for n in range(len(class_list)):
            detects.append([])
        
        if(indexes.__class__.__name__!= "tuple"):
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                x1,y1 = x,y+int(self.b1)
                x2,y2 = x+w,y+h+int(self.b1)
                for cl in class_list:
                    if class_ids[i] == cl:
                        detects[class_list.index(cl)].append(np.array([x1,y1,x2,y2]))
        return detects
    
    def track(self,obj_new,obj_prev):
        '''
        input:
            obj_list_new [tuple] : (centrePoint [type: orderedDict],box [type: orderedDict])

            track = [id,x,y,x1,y1,x2,y2]
        '''
        ##Combine centre and box
        ##Get dict length    
        centre_new,box_new = obj_new
        centre_prev,box_prev = obj_prev

        new_keys = list(centre_new.keys())
        prev_keys = list(centre_prev.keys())

        obj_new_list = []
        obj_prev_list = []
        
        for k in new_keys:
            obj_new_list.append([k,centre_new[k][0],centre_new[k][1],box_new[k][0],box_new[k][1],box_new[k][2],box_new[k][3]])
        for k in prev_keys:    
            obj_prev_list.append([k,centre_prev[k][0],centre_prev[k][1],box_prev[k][0],box_prev[k][1],box_prev[k][2],box_prev[k][3]])

        #### tracks = [[[id_new,x,y,x1,y1,x2,y2],[id_prev,x,y,x1,y1,x2,y2]],[[id_new,x,y,x1,y1,x2,y2],[id_prev,x,y,x1,y1,x2,y2]],....]
        tracks = []

        for objNew in obj_new_list: 
            objPrev_temp = None
            for objPrev in obj_prev_list:
                if objNew[0]==objPrev[0]:
                    tracks.append([objNew,objPrev])
                    break
            else:
                tracks.append([objNew,[99,0,0,0,0,0,0]])
        return tracks
    
    def orientation(self,p,q,r):
        '''
        par:
            p : [x,y]
            q : [x,y]
            r : [x,y]
        return:
            0 for colinear
            1 for clockwise
            2 for anticlockwise
        '''
        val = (p[1]-q[1])*(q[0]-r[0])-(p[0]-q[0])*(q[1]-r[1])

        if val > 0:
            return 1
        elif val<0:
            return 2
        else:
            return 0

    def onSegment(self,p,q,r):
        #is q lies on line pr?
        if q[0] <= min(p[0],r[0]) and q[0] >= max(p[0],r[0]) and q[1] <= max(p[1],r[1]) and q[1] >= min(p[1],r[1]):
            return True
        return False

    def intersectCheck(self,line1,line2):
        '''
        par:
            line1 : [[x1,y1],[x2,y2]]
            line2 : [[x1,y1],[x2,y2]]
        return:
            true for intersect line
            false for no intersect
        '''
        p1,q1 = line1
        p2,q2 = line2
        o1 = self.orientation(p1,q1,p2)
        o2 = self.orientation(p1,q1,q2)
        o3 = self.orientation(p2,q2,p1)
        o4 = self.orientation(p2,q2,q1)

        # General Case
        if (o1 != o2) and (o3 != o4):
            return True
        # Spesial Case
        if o1 == 0 and self.onSegment(p1,p2,q1):
            return True
        if o2 == 0 and self.onSegment(p1,p2,q1):
            return True
        if o3 == 0 and self.onSegment(p2,p1,q2):
            return True
        if o4 == 0 and self.onSegment(p2,q1,q2):
            return True
        return False

    def run_detector(self):
        
        ret,frame = self.cap.read()
        tracks = []
        objects = []
        if ret != False:
            
            start_time = time.time()

            self.frame += 1

            self.ui.progressBar.setValue(int(self.frame*100/self.frameCount))

            roi = frame[int(self.b1):int(self.d1)]

            boxes,confidences,class_ids = self.detect_frame(roi,int(self.ui.objthreshSpinbox.value())/100)
            
            indexes = cv2.dnn.NMSBoxes(boxes,confidences,int(self.ui.objthreshSpinbox.value())/100,int(self.ui.nmsthreshSpinbox.value())/100)
            
            detects = self.convert_detect(boxes,class_ids,indexes,self.class_list_detect)
            
            if self.frame == 1:
                for cld in range(len(self.class_list_detect)):
                    self.objects_temp.append((collections.OrderedDict(),collections.OrderedDict()))

            for cld in range(len(self.class_list_detect)):
                self.ct[cld].update(detects[cld])
                objects.append(self.ct[cld].update(detects[cld]))
                tracks.append(self.track(objects[cld],self.objects_temp[cld]))
            
            if self.frame%5 == 0:
                self.objects_temp = deepcopy(objects)
            source_frame = frame.copy()
            

            ##Count intersect line
            
                    
            '''
            Draw all detection,line,..
            '''
            cv2.line(frame,(int(self.p1),int(self.q1)),(int(self.p2),int(self.q2)),(0,0,255),2)
            cv2.line(frame,(int(self.a1),int(self.b1)),(int(self.a2),int(self.b2)),(0,211,255),2)
            cv2.line(frame,(int(self.c1),int(self.d1)),(int(self.c2),int(self.d2)),(0,211,255),2)

            line_count = False
            for track in range(len(tracks)):
                for t in tracks[track]:
                    new_box,old_box = t
                    cv2.rectangle(frame,(new_box[3],new_box[4]),(new_box[5],new_box[6]),self.colors[track],2)
                    (label_width, label_height), baseline = cv2.getTextSize(str(new_box[0])+"-"+self.class_list[track], self.font, 2, 2)
                    cv2.rectangle(frame,(new_box[3],new_box[4]),(new_box[3]+label_width,new_box[4]-label_height),self.colors[track],-1)
                    cv2.putText(frame,str(new_box[0])+"-"+self.class_list[track],(new_box[3],new_box[4]),self.font,2,(255,255,255),2)
                    
                    if old_box[0] != 99:
                        cv2.line(frame,(new_box[1],new_box[2]),(old_box[1],old_box[2]),(0,255,255),2)
                    
                    cv2.line(frame,(new_box[3],new_box[4]),(new_box[5],new_box[6]),self.colors[track],2)
                    if self.intersectCheck([[new_box[3],new_box[4]],[new_box[5],new_box[6]]],[[self.p1,self.q1],[self.p2,self.q2]]) and not (new_box[0] in self.counter_id_mem[track]):
                        line_count = True
                        cv2.rectangle(frame,(new_box[3],new_box[4]),(new_box[5],new_box[6]),self.colors[track],10)
                        self.counter[track]+=1
                        self.counter_id_mem[track].append(new_box[0])
            if line_count:
                cv2.line(frame,(int(self.p1),int(self.q1)),(int(self.p2),int(self.q2)),(0,255,255),3)
            fps = round(1/(time.time() - start_time),2)
            label = [str(self.video_source.split("/")[-1]),"FPS: "+str(fps),"frame no: "+str(self.frame),"Pothole: "+str(self.counter[0]),"Alligator Crack: "+str(self.counter[1]),"Lateral Crack: "+str(self.counter[2]),"Longitudinal Crack: "+str(self.counter[3])]
            
            label_size = []
            for l in label:
                width,height = cv2.getTextSize(l,cv2.FONT_HERSHEY_DUPLEX,0.8,2)[0]
                label_size.append([width,height])

            box_width = max([i[0] for i in label_size]) + 20
            box_height = (15*len(label))+20+sum([i[1] for i in label_size])
        
            cv2.rectangle(frame,(0,0),(box_width,box_height),(32,32,32),-1)
            
            for l in label:
                y = 20    
                i = label.index(l)
                cv2.putText(frame,l,(10,y+15*i+sum([i[1] for i in label_size[:i+1]])),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
            
            
            
            
            self.ui.pothole_count.setText(str(self.counter[0]))       
            self.ui.alligator_count.setText(str(self.counter[1]))
            self.ui.linear_count.setText(str(self.counter[2]))
            self.ui.longitudinal_count.setText(str(self.counter[3]))
            self.out_vid.write(frame)
            self.setMainScreen(frame)
            self.setSourceScreen(source_frame)
            
        else:
            print(str(self.video_source.split("/")[-1]))
            print(time.time()-self.start_time,"second")
            print(int(self.frameCount)/(time.time()-self.start_time),"FPS")
            self.frame = 0
            self.timer.stop()
            self.cap.release()
            self.out_vid.release()
            self.ui.progressBar.hide()
            self.ui.runButton.setEnabled(True)
            self.ui.counterlineSpinbox.setEnabled(True)
            self.ui.objthreshSpinbox.setEnabled(True)
            self.ui.nmsthreshSpinbox.setEnabled(True)
            self.ui.maxdiasppearedSpinbox.setEnabled(True)
            self.ui.centroidthreshSpinbox.setEnabled(True)


    def setMainScreen(self,frame):
        heightScreen = self.ui.main_screen.size().height()
        widthScreen = self.ui.main_screen.size().width()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape

        ratioA = widthScreen/width
        ratioB = heightScreen/height

        if ratioA<ratioB:
            self.ratio = ratioA
        else:
            self.ratio = ratioB

        frame = cv2.resize(frame,(int(width*self.ratio),int(height*self.ratio)))
        

        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.ui.main_screen.setPixmap(QPixmap.fromImage(qImg))

    def setSourceScreen(self,frame):
        heightScreen = self.ui.source_screen.size().height()
        widthScreen = self.ui.source_screen.size().width()
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape

        ratioA = widthScreen/width
        ratioB = heightScreen/height

        if ratioA<ratioB:
            ratio = ratioA
        else:
            ratio = ratioB

        frame = cv2.resize(frame,(int(width*ratio),int(height*ratio)))


        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.ui.source_screen.setPixmap(QPixmap.fromImage(qImg))

    def exit_program(self):
        sys.exit()

    def about(self):
        QMessageBox.about(self, "About This Program", 
        "<p align='center'>"
        "<b>Thanks for:</b><br>"
        "o Adrian Rosebrock, PhD - pyimagesearch.com<br>"
        "o pjreddie - Darknet<br>"
        "<i>and all packages who support this program</i><br><br>"
        "<b>Road Damage Recognition</b><br>"
        "Alvaro Basily | 2020</p>")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = main()
    window.show()

    sys.exit(app.exec_())