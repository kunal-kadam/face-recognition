# Importing required library
from math import nan
import cv2
import threading
import time
import os

from get_base_frame import get_frame
from face_recognition import predict_face
from constants import *

get_frame('Hello Photo please!', BASE_IMAGE)

def collectFrames():
    COLLECTED_IMAGES = []
    for i in range(FRAMES_NUMBER):
        suc, frame = cap.read()
        if not suc : continue
        imgname = os.path.join(FOLDER_PATH, f'image_{i}.jpg')
        cv2.imwrite(imgname, frame)
        COLLECTED_IMAGES.append('images'+str(i)+'.jpg')
        time.sleep(FRAME_INTERVAL)
    return len(COLLECTED_IMAGES)

def myPeriodicFunction():
    try:
        collectedLength = collectFrames()
        if collectedLength > 0:
            try:
                result = predict_face()
            except:
                result = nan
                print("No Face found in the frame")
        else : result = nan
        print(f"Result: {result}: \t Status: {result < THRESHOLD}")
    except:
        print(f"Failed to recognize face")

open_threads = []
def startTimer():
    t = threading.Timer(TIME_INTERVAL, startTimer)
    t.start()
    myPeriodicFunction()
    open_threads.append(t)

# cap = cv2.VideoCapture('./data/nthu8/validate/noglasses/022_noglasses_mix.mp4')
cap = cv2.VideoCapture(0)
startTimer()

while True : 
    suc, frame = cap.read()
    if not suc :
        break
    cv2.imshow("webcamp", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()

for t in open_threads:
    if t.is_alive(): t.cancel()