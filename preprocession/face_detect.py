# importing library
import numpy as np
import cv2
import dlib
from imutils import face_utils
import pandas as pd
import os
from tqdm import tqdm

# loading data
dataframe_list = pd.read_csv('./data/nthu8/train/sleepy/sleep.csv', index_col=0)
print(dataframe_list.shape)
print(dataframe_list.head(4))

# constants
TRAIN_SIZE = dataframe_list.shape[0]
dataframe_temp = []

# models
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# for v in tqdm(range(0, TRAIN_SIZE)):
for i in os.listdir('./data/nthu8/train/frames/Yawning'):
    # video
    cam = cv2.imread('./data/nthu8/train/frames/Yawning/'+i)
    print('./data/nthu8/train/frames/Yawning/'+i)
    data_tuple = []
    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray)

    for face in faces:
        #detect landmarks
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)
        shape = shape.reshape(-1).tolist()
        data_tuple.append(i)
        data_tuple.extend(shape)
        data_tuple.append(1)
    if(len(data_tuple) == 138):
        dataframe_temp.append(data_tuple)
        data_tuple = []
    # label
    # file=open('./data/nthu8/train/sleepy/label/'+dataframe_list.iloc[v].label, 'r')
    # print('./data/nthu8/train/sleepy/label/'+dataframe_list.iloc[v].label)
    # data = file.read()
    # data = np.array(list(data)).astype(int)

    
    # data_tuple = []
    # j = -1
    # count = 0
    # while True : 
    #     suc, frame = cam.read()
        
    #     if not suc : 
    #         break
    #     count += 1
    #     j += 1
    #     if(count%5 != 0): continue
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_model(gray)
        

    #     for face in faces:
    #         #detect landmarks
    #         shapes = landmark_model(gray, face)
    #         shape = face_utils.shape_to_np(shapes)
    #         shape = shape.reshape(-1).tolist()
    #         data_tuple.append(dataframe_list.iloc[v].video)
    #         data_tuple.append(j)
    #         data_tuple.extend(shape)
    #         data_tuple.append(data[j])
    #         # print(len(data_tuple))
    #     if(len(data_tuple) == 139):
    #         dataframe_temp.append(data_tuple)
    #     data_tuple = []
    #     # cv2.imshow("webcamp", frame)
    #     # if cv2.waitKey(1) & 0xFF == ord('q') : 
    #     #     break

    # print(f"TRAIN VIDEO {v}")
    # cv2.destroyAllWindows()
    #52

columns = ['filename',]
for i in range(68):
    columns.append('x'+str(i))
    columns.append('y'+str(i))

columns.append('label')
dataframe = pd.DataFrame(dataframe_temp,columns=columns)
# dataframe.append()
dataframe.to_csv('./data/nthu8/train/yawning/yawn_mouth_landmark_1.csv')