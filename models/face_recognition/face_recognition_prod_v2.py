# Importing required library
import cv2
import time
from schedule import every, repeat

import sys
sys.path.insert(0, './models')
from get_base_frame_v1 import get_frame, detect_face
from constants import *
from authenticate import encrypt_decrypt
from schedular import schedular
from sockets import socket


# Get verification frames
get_frame('Front Face Verification', 1, 'profile')
get_frame('Front Face Reverification', 2, 'reverifyprofile')


# Schedule a function in the backend thread for ever interval time
@repeat(every(FRAME_INTERVAL).seconds)
def collect_frames():
    suc, frame = cam.read() # capture the current frame
    if not suc : return 0
    t = time.time() 
    result = detect_face(frame, f"frames/frame{t}.txt") # detect face and store face embeddings in encrypted text file
    if result == 1:
        socket.send_file_v2(f"{FOLDER_PATH}frame{t}.txt") # send file to the server via socket connection
        time.sleep(3)
        filename = socket.receive_file_v2(f"frame{t}.txt", RESULT_PATH) # get the result file from the server via socket connection
        result = encrypt_decrypt.decrypt_client_side(filename) # decrypt the result
        print("Person face recognition score: ", result)
        encrypt_decrypt.remove_decrypted_files(f"{FOLDER_PATH}frame{t}.txt", f"{RESULT_PATH}result_frame{t}.txt")
    else:
        pass # person not found in the frame


# Start testing
cam = cv2.VideoCapture(0)


# set the background thread
stop_run_continuously_frames = schedular.run_continuously(FRAME_INTERVAL)


while True : 
    suc, frame = cam.read()
    if not suc :
        break
    cv2.imshow("webcamp", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cam.release()
cv2.destroyAllWindows()

stop_run_continuously_frames.set() # stop the background thread