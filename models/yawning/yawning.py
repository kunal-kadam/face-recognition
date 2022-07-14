# Importing library
import numpy as np
import json, cv2

import dlib
from imutils import face_utils

import sys
sys.path.insert(0, './models')
from utils import angle_between
import models
import play_mp3


# Defining constants
MAR_THRESH = 20.00
MAR_CONSEC_FRAMES = 12
COUNTER = 0


# Models
face_model = dlib.get_frontal_face_detector() # Detecting face
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') # Getting landmarks
with open('./models/logistic_regression_nthu.txt', 'r') as file:
    json_text = json.load(file)
model_test = models.deserialize_logistic_regression(json_text)


# Calculate distance between upper lip and lower lip
def lip_distance(shape):
    '''
    **Input
    mouth -> np.array (48-68)landmarks dlib
    **Output
    distance -> float
    '''
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# ML model the predicts the person is yawning or not base on the angles
# made on left and right side of mouth
def logistic_regression_model(shape):
    '''
    **Input
    shape -> np.array (48-68)landmarks dlib
    **Output
    result -> float, 0(Active) 1(Yawning)
    '''
    distance = lip_distance(shape=shape)
    loa = angle_between(shape[58], shape[48], shape[50])
    roa = angle_between(shape[52], shape[54], shape[56])

    return distance, model_test.predict(np.array([[loa, roa]]))[0]

# Testing frame
cam = cv2.VideoCapture('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')
print('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')

# Testing frame labels
file=open('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt', 'r')
print('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt')
data = file.read() 
data = np.array(list(data)).astype(int)
j = -1

# Starting testing
while True : 
    suc, frame = cam.read()
    
    if not suc :
        cv2.destroyAllWindows()
        break

    # Convert images of any kind to gray scale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray)

    j += 1
    for face in faces:
        # detect landmarks
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)
        print(data[j])

        cv2.imshow("webcamp", frame)

        distance, result_lr = logistic_regression_model(shape=shape)

        if distance > MAR_THRESH and result_lr == 1.0:
            print("Yawning!")
            COUNTER += 1
            if COUNTER >= MAR_CONSEC_FRAMES:
                print("Hello Good Morning :)")
                COUNTER = -5 # to avoid to much repetion after every alert message
                play_mp3.play()
        else:
            if(COUNTER > 0):
                COUNTER -= 1 # person is active in this frame
    if cv2.waitKey(1) & 0xFF == ord('q') : 
            break

cam.release()
cv2.destroyAllWindows()