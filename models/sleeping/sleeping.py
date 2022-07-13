# Importing required library
import numpy as np
import json, cv2

from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

import playsound, random

import sys
sys.path.insert(0, './models')
from utils import angle_between
import models
import play_mp3


# Defining constants
EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 40
COUNTER = 0


# Loading pretrained models
face_model = dlib.get_frontal_face_detector() # Detecting face
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') # Getting landmarks
with open('./models/decision_tree_google.txt', 'r') as file:
    json_text = json.load(file)
model_test = models.deserialize_decision_tree(json_text)


# Calculate distance between upper and lower eye lid
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mean of the distances of left and right eye
def final_ear(shape):
    '''
    **Input
    shape -> np.array (48-68)landmarks dlib
    **Output
    ear -> float
    '''
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear, leftEAR, rightEAR

# ML model that predicts the sleepiness of the person in frame
def decision_tree_model(shape):
    '''
    **Input
    shape -> np.array (48-68)landmarks dlib
    **Output
    result -> distance -> float, 0(Active) 1(Sleepy)
    '''
    ear, lar, rar = final_ear(shape=shape)
    lia_1 = angle_between(shape[41], shape[36], shape[37])
    lia_2 = angle_between(shape[38], shape[39], shape[40])
    ria_1 = angle_between(shape[44], shape[45], shape[46])
    ria_2 = angle_between(shape[47], shape[42], shape[43])

    lia = (lia_1+lia_2)/2
    ria = (ria_1+ria_2)/2
    return ear, model_test.predict(np.array([[lia, ria, lar, rar]]))[0]

# Testing frame
cam = cv2.VideoCapture('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')
print('./data/nthu8/train/lauging/glasses_009_nonsleepyCombination.avi')

# Testing frame labels
file=open('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt', 'r')
print('./data/nthu8/train/lauging/label/glasses_009_nonsleepyCombination_mouth.txt')
data = file.read() 
data = np.array(list(data)).astype(int)
j = -1

# Start testing
while True : 
    suc, frame = cam.read()
    
    if not suc : 
        break
    
    # Convert images of any kind to gray scale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray) # detect face

    j += 1
    for face in faces:
        # detect landmarks
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)
        print(data[j])

        cv2.imshow("webcamp", frame)

        ear, dt_result = decision_tree_model(shape)
        
        if ear < EAR_THRESH and dt_result == 1.0:
            print("Sleeping!")
            COUNTER += 1
            if (COUNTER > EAR_CONSEC_FRAMES): # sleepy eyes are detected in many consecutive frames
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