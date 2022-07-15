import os, cv2
import pandas as pd
import numpy as np
import mediapipe
mp_face_mesh = mediapipe.solutions.face_mesh

data_frame = pd.read_csv('./data/nthu8/train/lauging/laugh.csv', index_col=0)
print(data_frame.shape)
print(data_frame.head())

data_dict = {'x0': [], 'y0': [], 'x4': [], 'y4': [], 'x61': [], 'y61': [], 'x292': [],
'y292': [], 'x289': [], 'y289': [], 'x17':[], 'y17': [], 'x50':[],
'y50':[], 'x280':[], 'y280': [], 'x48': [], 'y48': [], 'x206': [],
'y206': [], 'x426': [], 'y426': [], 'label': []}

for i in range(data_frame.shape[0]):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False)
    cam = cv2.VideoCapture('./data/nthu8/train/lauging/'+data_frame.iloc[i].video)
    print('./data/nthu8/train/lauging/'+data_frame.iloc[i].video)
    file=open('./data/nthu8/train/lauging/label/'+data_frame.iloc[i].label, 'r')
    print('./data/nthu8/train/lauging/label/'+data_frame.iloc[i].label)
    data = file.read()
    data = np.array(list(data)).astype(int)
    j = -1
    data_tuple = []

    while True:
        suc, frame = cam.read()
        if not suc : 
            break
        j += 1
        print(data[j])
        # cv2.imshow("webcamp", frame)


        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks == None: continue
        landmarks = results.multi_face_landmarks[0]
        for idx, landmark in enumerate(landmarks.landmark):
            if idx == 61 or idx == 292 or idx == 0 or idx == 17 or idx == 50 or idx == 280 or idx == 48 or idx == 4 or idx == 289 or idx == 206 or idx == 426:
                x = landmark.x
                y = landmark.y
                data_dict['x'+str(idx)].append(x)
                data_dict['y'+str(idx)].append(y)
        data_dict['label'].append(data[j])
        if cv2.waitKey(1) & 0xFF == ord('q') : 
            break
    # cv2.destroyAllWindows()
    cam.release()   

df = pd.DataFrame(data_dict)
df.to_csv('./data/nthu8/train/lauging/laugh_mouth_landmark.csv')