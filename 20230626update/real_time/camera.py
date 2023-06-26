import cv2
import torch
import numpy as np
      
#model = torch.load('yolov5s.pt')
#model.eval()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()

        results = model(image, size=640)
        img = np.squeeze(results.render()) 
        
        return cv2.imencode('.jpg', img)[1].tobytes()