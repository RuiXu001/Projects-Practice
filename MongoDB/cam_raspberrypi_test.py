from mongoengine import *
import matplotlib.pyplot as plt 
from datetime import datetime as dt
import pandas as pd
import datetime
import numpy as np
import time
import cv2

DB_URI = "mongodb+srv://"
connect(db="test",host=DB_URI)

class Cam(Document):
    meta = {'collection': 'cam1'}
    date_time = DateTimeField(default=dt.utcnow)
    cam_id = StringField(required=True)
    img_type = StringField(required=True)
    image = FileField(thumbnail_size=(150,150,False))

def upload_img(cam_id, img_type, img):
    img_str = np.array(img).tobytes() # convert to bytes
    record = Cam(
        cam_id = cam_id,
        img_type = img_type)
    record.image.put(img_str)
    record.save()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    print( cap.isOpened())
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame size {}'.format(size))
    start_time = time.time()
    interval = 60 # 60 seconds
    while True:
        success, img = cap.read()
        if time.time() - start_time > interval:
            #show_img(img)
            start_time += interval
            upload_img('RaspberryPi', 'clock_test', img)
            print('Uploading ', dt.now())
