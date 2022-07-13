import cv2

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL  
ORG = (20, 30)
FONTSCALE = 1
COLOR = (144, 70, 31)
THICKNESS = 2
IMAGE_PATH = './data/recognition/'
FOLDER_PATH = './data/recognition/frames/'
RESULT_PATH = './data/recognition/results/'
ERROR = 'Incorrect Image'

BASE_IMAGE = 'person.jpg'
BASE_FILENAME = 'profile'
FRAME_NO = 0
COLLECTED_IMAGES = []
FRAMES_NUMBER = 5
FRAME_INTERVAL = 5 #sec
TIME_INTERVAL = 30 #sec

MODEL_NAME = 'Facenet'
DISTANCE_METRIC = 'euclidean'
THRESHOLD = 100 # https://www.youtube.com/watch?v=i_MOwvhbLdI
DETECTOR_BACKEND = 'opencv'