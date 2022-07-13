# importing library
import numpy as np
import cv2
import dlib
import time 
from imutils import face_utils
import pandas as pd
from tqdm import tqdm

# loading data
dataframe = pd.read_csv('./data/nthu8/train/yawning/yawn.csv', index_col=0)
print(dataframe.shape)
print(dataframe.head(4))

# constants and initials
EPOCH = 5
TRAIN_SIZE = dataframe.shape[0]
ERROR_EXP = 0.001
MOR_LIMIT = 20.4195
LR = 0.05

# models
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# calculate error
def calculate_error(actual, predict):
  # actual 1D array, predict 1D array
  result = (((actual*-2)+1) - ((predict*-2)+1)) / 2
  exp = []
  a = ERROR_EXP
  for i in range(result.shape[0]-1):
    if result[i] != 0:
      if result[i-1] == result[i] or i == 0:
        exp.append(np.exp(a))
        a += ERROR_EXP
      elif result[i-1] != result[i]:
        a = ERROR_EXP
        exp.append(np.exp(a))
        a += ERROR_EXP
    else:
      exp.append(0)
      a = ERROR_EXP
    return np.sum(exp*result)

for i in tqdm(range(EPOCH)):
    for v in tqdm(range(52, TRAIN_SIZE)):
        # video
        cam = cv2.VideoCapture('./data/nthu8/train/yawning/'+dataframe.iloc[v].video)
        print('./data/nthu8/train/yawning/'+dataframe.iloc[v].video)

        # label
        file=open('./data/nthu8/train/yawning/label/'+dataframe.iloc[v].label, 'r')
        print('./data/nthu8/train/yawning/label/'+dataframe.iloc[v].label)
        data = file.read()
        data = np.array(list(data)).astype(int)
        predict_data = []
        actual_data = []
        j = 0
        while True : 
            suc, frame = cam.read()
            
            if not suc : 
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_model(gray)

            for face in faces:
              #detect landmarks
              shapes = landmark_model(gray, face)
              shape = face_utils.shape_to_np(shapes)

              distance = lip_distance(shape=shape)
              print(distance)
              if(distance > MOR_LIMIT): 
                  predict_data.append(1)
                  print(1, data[j])
                  actual_data.append(data[j])
              else: 
                  predict_data.append(0)
                  print(0, data[j])
                  actual_data.append(data[j])
              if(j < data.shape[0]-1): j += 1

            cv2.imshow("webcamp", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') : 
                break
        predict_data = np.array(predict_data)
        actual_data = np.array(actual_data)
        # error = calculate_error(actual_data, predict_data)
        error = np.sum(np.abs(actual_data-predict_data))/actual_data.shape[0]
        if(np.sum(actual_data) > np.sum(predict_data)): error = -error*ERROR_EXP
        MOR_LIMIT = MOR_LIMIT + ((LR/(i+1)) * error)

        print(f"EPOCH {i} : TRAIN VIDEO {v} : error {error} : MOR_LIMIT {MOR_LIMIT}")
        cv2.destroyAllWindows()
        #52