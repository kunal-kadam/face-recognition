# Importing required library
import numpy as np
import cv2, json

import mediapipe

import sys
sys.path.insert(0, './models')
from utils import angle_between
import models
import play_mp3


# Defining constants
LAR_CONSEC_FRAMES = 40
COUNTER = 0

# Loading pretrained models
mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False)
with open('./models/decision_tree_laughing_nthu.txt', 'r') as file:
    json_text = json.load(file)
model_test = models.deserialize_decision_tree(json_text)


# ML model that predicts the sleepiness of the person in frame
def decision_tree_model(shape):
    '''
    **Input
    shape -> np.array (468)landmarks mediapipe
    **Output
    result -> 0(Active) 1(Yawning) 2(Speaking)
    '''
    mla = angle_between((shape[17].x, shape[17].y), (shape[61].x, shape[61].y), (shape[0].x, shape[0].y))
    mua = angle_between((shape[61].x, shape[61].y), (shape[0].x, shape[0].y), (shape[292].x, shape[292].y))
    nta = angle_between((shape[48].x, shape[48].y), (shape[4].x, shape[4].y), (shape[289].x, shape[289].y))
    cna = angle_between((shape[206].x, shape[206].y), (shape[4].x, shape[4].y), (shape[426].x, shape[426].y))
    mna = angle_between((shape[61].x, shape[61].y), (shape[4].x, shape[4].y), (shape[292].x, shape[292].y))
    cra = angle_between((shape[289].x, shape[289].y), (shape[280].x, shape[280].y), (shape[292].x, shape[292].y))
    cria = angle_between((shape[289].x, shape[289].y), (shape[426].x, shape[426].y), (shape[292].x, shape[292].y))

    return  model_test.predict(np.array([[mla, mua, nta, cna, mna, cra, cria]]))[0]


# Testing frame
# cam = cv2.VideoCapture('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')
cam = cv2.VideoCapture(0)
print('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')

# Testing frame labels data\nthu8\train\yawning\noglasses_002_yawning.avi
# file=open('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt', 'r')
# print('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt')
# data = file.read() 
# data = np.array(list(data)).astype(int)
# j = -1

# Start testing
while True : 
    suc, frame = cam.read()
    
    if not suc :
        cv2.destroyAllWindows()
        break

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks == None: continue

    # j += 1
    for face in results.multi_face_landmarks:
        dt_result = decision_tree_model(face.landmark)
        # print(data[j])

        if dt_result == 2.0:
            print('Speaking!')
            COUNTER += 1
            if (COUNTER > LAR_CONSEC_FRAMES):
                print("Hello keep quite :>")
                COUNTER = -5
                play_mp3.play("speak")
        else:
            if(COUNTER > 0):
                COUNTER -= 1 # person is active in this frame
    cv2.imshow("webcamp", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') : 
        break
        