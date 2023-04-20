import cv2
import torch
import numpy as np
      
# model = torch.load('.\best_20.pt')
# model.eval()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
#        self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # image=cv2.flip(image, -1)
        results = model(image)
        #a = np.squeeze(results.render())

        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)

        import PIL.Image as pilimg
        import numpy as np

        image = pilimg.open('test.jpg')

        image.show()
        pix = np.array(image)

        print(jpeg.tobytes())
        return pix.tobytes()