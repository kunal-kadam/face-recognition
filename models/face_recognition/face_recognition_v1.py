from deepface import DeepFace
from constants import *
import os

def predict_face():
    image_path = os.path.join(IMAGE_PATH, BASE_IMAGE)
    df = DeepFace.find(image_path, FOLDER_PATH, model_name=MODEL_NAME, distance_metric=DISTANCE_METRIC)
    for f in os.listdir(FOLDER_PATH):
        os.remove(os.path.join(FOLDER_PATH, f))
    print(df)
    return df.Facenet_euclidean.mean()

if __name__ == "__main__":
    res = predict_face(BASE_IMAGE)
    print(res)