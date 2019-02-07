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

    def get_output_image(self, frame):
        if self.flip:
            flipped_frame = cv2.flip(frame, 0)
            return cv2.imencode('.jpg', flipped_frame)
        return cv2.imencode('.jpg', frame)

    def save_frame(self):
        ret, frame = self.vc.read()
        if self.flip:
            flipped_frame = cv2.flip(frame, 0)
            return self.out.write(flipped_frame)
        return self.out.write(frame)

    def get_frame(self):
        ret, frame = self.vc.read()
        ret, image = self.get_output_image(frame)
        return image.tobytes()
