# Importing required library
import cv2
import threading
import time

import sys
sys.path.insert(0, './models')
from get_base_frame_v1 import get_frame, detect_face
from constants import *
from authenticate import encrypt_decrypt

# get photos uploaded for verifiaction and reverification
get_frame('Hello Photo please!', 2)


def collectFrames():
    COLLECTED_IMAGES = []
    for i in range(FRAMES_NUMBER):
        suc, frame = cap.read()
        if not suc : continue
        result = detect_face(frame, f"frames/frame{i}.txt")
        if(result == 1): COLLECTED_IMAGES.append(f'frame{i}.txt')
        time.sleep(FRAME_INTERVAL)
    return len(COLLECTED_IMAGES)

def myPeriodicFunction():
    collectedLength = collectFrames()
    print(collectedLength)
    if collectedLength > 0:
        encrypt_decrypt.server_calculation_v1(FOLDER_PATH, IMAGE_PATH, RESULT_PATH)
        encrypt_decrypt.decrypt_client_side_v1(RESULT_PATH)

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