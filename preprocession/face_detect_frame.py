from cv2 import imread
import numpy as np
import cv2
import dlib
from imutils import face_utils
import pandas as pd
from tqdm import tqdm
import os
dataframe_temp = []
data_tuple = []

# models
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

output_path = 'C:/Users/Kunal Kadam/Desktop/Intel/data/nthu8/train/frames'
frame_type = '/active'
for i in os.listdir(output_path+frame_type):
    print(i)
    frame = imread('./data/nthu8/train/frames'+frame_type+'/'+i)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray)

    for face in faces:
        #detect landmarks
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)
        shape = shape.reshape(-1).tolist()
        data_tuple.append(i)
        data_tuple.extend(shape)
        data_tuple.append(0)
        # print(len(data_tuple))
    if(len(data_tuple) == 138):
        dataframe_temp.append(data_tuple)
    data_tuple = []

    columns = ['filename']
for i in range(68):
    columns.append('x'+str(i))
    columns.append('y'+str(i))

columns.append('label')
dataframe = pd.DataFrame(dataframe_temp,columns=columns)
# dataframe.append()
dataframe.to_csv('./data/nthu8/train/yawning/active_frame_landmark.csv')