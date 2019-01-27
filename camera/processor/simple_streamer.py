from imutils.video.pivideostream import PiVideoStream
import time
import datetime
import numpy as np
import cv2


class SimpleStreamer(object):
    def __init__(self, flip = False):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        # Video Capture
        try:
            self.vc = cv2.VideoCapture(0)
        except:
            print(self.vc)
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        self.out.release()
        self.vc.release()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
