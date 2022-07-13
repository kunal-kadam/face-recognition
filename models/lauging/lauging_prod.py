# Importing library
import numpy as np
import cv2
import dlib
from imutils import face_utils
from datetime import datetime
import os
import re
import json
from sklearn.linear_model import LogisticRegression

# Defining constants
MOR_LIMIT = 20.568
MOR_THERSHOLD = 7
MOR_FRAMES = 20
MOR_BUFFER = 2
SAVE_FRAME_THRESHOLD = 20
IMAGES_PATH = './data/yawning_image/'

# Models
face_model = dlib.get_frontal_face_detector() # Detecting face
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') # Getting landmarks
with open('./models/laugh_mor_classifier.txt', 'r') as file:
    json_text = json.load(file)
json_text
model_test = LogisticRegression()
model_test.coef_ = np.array(json_text['coef'])
model_test.intercept_ = np.array(json_text['intercept'])
model_test.classes_ = np.array(json_text['classes'])

def lip_distance2(shape):
    distance = lip_distance(shape=shape)
    mouth_width = shape[54][0] - shape[48][0]
    outer_mouth_1 = shape[50][1] - shape[58][1]
    outer_mouth_2 = shape[51][1] - shape[57][1]
    outer_mouth_3 = shape[52][1] - shape[56][1]
    inner_mouth_1 = shape[61][1] - shape[67][1]
    inner_mouth_2 = shape[62][1] - shape[66][1]
    inner_mouth_3 = shape[63][1] - shape[65][1]
    return model_test.predict(np.array([[distance, mouth_width, outer_mouth_1, 
        outer_mouth_2, outer_mouth_3, inner_mouth_1, inner_mouth_2, inner_mouth_3]]))[0]

# Calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Get current time
def get_current_frame_name():
    return re.sub(r"[^0-9]", '_', str(datetime.now())) + '.jpg'

# cam = cv2.VideoCapture('./data/nthu8/train/yawning/noglasses_001_yawning.avi')
cam = cv2.VideoCapture('./data/nthu8/train/lauging/noglasses_013_nonsleepyCombination.avi')
# cam = cv2.VideoCapture(0)

counter = 0
save_counter = 0

while True : 
    suc, frame = cam.read()
    
    if not suc :
        cv2.destroyAllWindows()
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray)

    for face in faces:
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)
        # distance = lip_distance(shape=shape)
        test = lip_distance2(shape=shape)
        print(test)
        cv2.imshow("webcamp", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') : 
            break
        # if(distance > MOR_LIMIT):
        if(test > 0):
            counter += 1
            if(counter > MOR_THERSHOLD and save_counter < SAVE_FRAME_THRESHOLD):
                # save frame
                imgname = os.path.join(IMAGES_PATH, get_current_frame_name())
                cv2.imwrite(imgname, frame)
                print('save frame')
                save_counter += 1
            else: pass	
        else:
            if(MOR_BUFFER == 0):
                # set to initial state
                counter = 0
                save_counter = 0
                MOR_BUFFER = 2
            elif(counter > MOR_LIMIT):
                MOR_BUFFER -= 1
            else: pass

    
