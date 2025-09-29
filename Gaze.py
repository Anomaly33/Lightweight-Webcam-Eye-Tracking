
import cv2
import queue
import threading
from utils import get_config
import mediapipe as mp


# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")


class Detector:
    def __init__(
        self,
        output_size,
        show_stream=False,
        show_markers=False,
        show_output=False,
        gpu=0
    ):
        print("Starting face detector...")
        self.output_size = output_size
        self.show_stream = show_stream
        self.show_output = show_output
        self.show_markers = show_markers
        self.detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.capture = cv2.VideoCapture(0)
        _,frame = self.capture.read()
        shape = frame.shape
        self.frame_h = shape[0]
        self.frame_w = shape[1]
        self.scalex = 0.8
        self.scaley = 0.8
        self.boxx = 1-self.scalex
        self.boxy = 1-self.scaley
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        self.out_of_box=0
        t.start()
        self.make_in_box()


    def _reader(self):
        while True:            
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def get_frame(self):
        
        if self.out_of_box>30:
            self.make_in_box()

        frame = self.q.get()
        self.frame_h, self.frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.detector.process(rgb_frame)   
        if not output.multi_face_landmarks:
            return []
             
        landmark_points = output.multi_face_landmarks
        landmarks = landmark_points[0].landmark
        if not self.face_in_box(landmarks):
            self.out_of_box+=1
            return []
        
        points=[]
 
        for i in range(len(landmarks)):
            landmark={'x':landmarks[i].x,\
                        'y':landmarks[i].y,\
                        'z':landmarks[i].z,}
            for ax in ['x','y','z']:
                points.append(landmark[ax])
                
        if len(points)<478*3:
            return []
        return points
        
    def face_in_box(self,landmarks):
        if landmarks[152].y*self.frame_h > (1-self.boxy)*self.frame_h:
            return False
        if landmarks[10].y*self.frame_h < (self.boxy)*self.frame_h:
            return False
        if landmarks[254].x*self.frame_w < (self.boxx)*self.frame_w:
            return False
        if landmarks[454].x*self.frame_w > (1-self.boxx)*self.frame_w:
            return False
        self.out_of_box=0
        return True
    
    def make_in_box(self):
        in_box=False
        while not in_box:    
            frame = self.q.get()
            self.frame_h, self.frame_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.detector.process(rgb_frame)   
            if not output.multi_face_landmarks:
                continue
             
            landmark_points = output.multi_face_landmarks
            landmarks = landmark_points[0].landmark
            in_box=self.face_in_box(landmarks)
            cv2.rectangle(frame,(int(self.frame_w*self.boxx),int(self.frame_h*self.boxy)),(int(self.frame_w*(1-self.boxx)),int(self.frame_h*(1-self.boxy))),(0,255,0),3)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
        self.out_of_box=0
        cv2.destroyWindow('frame')
            
    def close(self):
        print("Closing face detector...")
        self.capture.release()
        cv2.destroyAllWindows()


