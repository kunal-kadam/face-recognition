import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def get_landmarks_dlib(frame):
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

    faces = face_detector(frame)

    landmark_tuple = []
    for k, d in enumerate(faces):
        landmarks = landmark_detector(frame, d)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    return landmark_tuple
img_path = './data/profile_image/kk1.jpg'

img = dlib.load_rgb_image(img_path)
lm = get_landmarks_dlib(img)
lm = np.array(lm)
up_lip = lm[48:55]
top_mean = np.mean(up_lip, axis=0)

cv2.drawContours(img, [up_lip], -1, (0, 165, 255), thickness=1)
cv2.imshow("webcamp", img)

cv2.waitKey(0)
cv2.destroyAllWindows()