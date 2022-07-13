import cv2
import os
from constants import *

def get_frame(text, image_name):
    # cam = cv2.VideoCapture('./data/nthu8/validate/noglasses/022_noglasses_mix.mp4')
    cam = cv2.VideoCapture(0)
    while True:
        while True :
            suc, frame = cam.read()
            
            if not suc :
                cv2.destroyAllWindows()
                break
            cv2.putText(frame, text, ORG, FONT, FONTSCALE, COLOR, THICKNESS)

            cv2.imshow("webcamp", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') : 
                imgname = os.path.join(IMAGE_PATH, image_name)
                cv2.imwrite(imgname, frame)
                break
        cv2.destroyAllWindows()

        image = cv2.imread(os.path.join(IMAGE_PATH, image_name))
        cv2.imshow("selected", image)
        if cv2.waitKey(0) & 0xFF == ord('e') : 
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(0) & 0xFF == ord('r') :
            cv2.destroyAllWindows()
            continue

if __name__ == '__main__':
    get_frame('Front Face Image', 'person2.jpg')