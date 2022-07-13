from asyncio import constants
from deepface import DeepFace
import math
import sys
sys.path.insert(0, './models')
from authenticate import encrypt_decrypt
from constants import * 
# First time at uploading data
profile_path = "./data/profile_image/kk1.jpg"

# At every query call update data
testing_frame_path = "./data/profile_image/jj.jpg"

# Convert face to vector embeddings
img1_embedding = DeepFace.represent(img_path = profile_path, detector_backend = DETECTOR_BACKEND, model_name = MODEL_NAME)
img2_embedding = DeepFace.represent(img_path = testing_frame_path, detector_backend = DETECTOR_BACKEND, model_name = MODEL_NAME) 

# Encrypt the data for privacy
encrypt_decrypt.encrypt_client_side(img_embedding=img1_embedding, file_name="profile.txt")
encrypt_decrypt.encrypt_client_side(img_embedding=img2_embedding, file_name="frame05.txt")

# Send to server and confirm the person 
encrypt_decrypt.server_calculation("frame05.txt")

response = encrypt_decrypt.decrypt_client_side("result.txt")
response = math.sqrt(response[0]) 

if (response < 9) == True:
    print("Person is same: "+ str(response))
else:
    print("Person is different: "+ str(response))

# 30 sec