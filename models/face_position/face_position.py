# Importing required library
import cv2
import mediapipe as mp
import numpy as np


# Loading pretrained models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.5,  max_num_faces=1)


# Start testing
cam = cv2.VideoCapture(0)

while True:
    suc, frame = cam.read()

    if not suc : 
        break

    # Flip the image for selfi-view and then convert the color space from BGR to RGB
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    
    # Get the face mesh
    results = face_mesh.process(frame)
    
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # 33 - left eye
                # 263 - right eye
                # 1 - nose tip
                # 61 - left mouth
                # 291 - right mouth
                # 199 - chin bottom
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y]) # Get the 2D coordinates
                    face_3d.append([x, y, lm.z]) # Get the 3D coordinates       
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Setting camera matrix to remove image distorsion
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)


            # Solve PnP (Perspective and Point problem)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          
            # Face position 
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 15:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            print(f"Position {text}: ({'%.2f'%x}, {'%.2f'%y}, {'%.2f'%z})")

    cv2.imshow('webcamp', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()