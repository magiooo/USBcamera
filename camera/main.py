from flask import Flask, render_template, Response
# from processor.simple_streamer import SimpleStreamer as VideoCamera
# from processor.pedestrian_detector import PedestrianDetector as VideoCamera
# from processor.motion_detector import MotionDetector as VideoCamera
# from processor.qr_detector import QRDetector as VideoCamera
# from processor.face_detector import FaceDetector as VideoCamera
from processor.person_detector import PersonDetector as VideoCamera

import time
import threading

video_camera = VideoCamera(flip=False)


app = Flask(__name__)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save')
def save():
    while True:
        video_camera.save_frame()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
