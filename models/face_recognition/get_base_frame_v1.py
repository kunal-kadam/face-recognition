# Importing required library
import cv2
import os
from deepface import DeepFace
import concurrent.futures

import sys
sys.path.insert(0, './models')
from authenticate import encrypt_decrypt
from constants import *
from sockets import socket


# detecting the face in the passed frame and storing the encrypted embeddings
def detect_face(image, filename):
    try:
        obj = DeepFace.represent(img_path = image, detector_backend = DETECTOR_BACKEND, model_name = MODEL_NAME)
        filename = os.path.join(IMAGE_PATH, filename)
        encrypt_decrypt.encrypt_client_side(img_embedding=obj, file_name=filename)
        return 1
    except:
        print("Face could not be detected")
        return 0


# frames that can be collected for verification purpose
def get_frame(text, cnt, base_file=BASE_FILENAME):
    cam = cv2.VideoCapture(0)
    i = 0
    while (i != cnt):
        while True :
            suc, frame = cam.read()
            
            if not suc :
                cv2.destroyAllWindows()
                break

            cv2.putText(frame, text + f" [{i}|{cnt}] ", ORG, FONT, FONTSCALE, COLOR, THICKNESS)
            cv2.imshow("webcamp", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') : 
                with concurrent.futures.ThreadPoolExecutor() as executor: # running a concurrent thread for detecting the face in frame
                    f1 = executor.submit(detect_face, frame, base_file+str(i)+'.txt')
                    if f1.result() == 0: 
                        cv2.putText(frame, ERROR + f" [{i}|{cnt}] ", ORG, FONT, FONTSCALE, COLOR, THICKNESS)
                        continue
                    else: break
        cv2.destroyAllWindows()

        cv2.imshow("selected", frame)
        if cv2.waitKey(0) & 0xFF == ord('e') : # pressing 'e' will select the image
            cv2.destroyAllWindows()
            socket.send_file_v2(IMAGE_PATH+base_file+str(i)+'.txt') # sending the image to the server via socket connection
            i +=1
        elif cv2.waitKey(0) & 0xFF == ord('r') : # pressing 'rr' will retake the image
            cv2.destroyAllWindows()
            continue

if __name__ == '__main__':
    get_frame('Front Face Image', 3)