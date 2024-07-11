#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
# extract hand landmarks
import mediapipe as mp


import  time 


import joblib






svm_model_path =  r'Custom_models\keypoint_classifier\svm_face_model.pkl'

keypoint_classifier_labels = ["happy","Angry","Neutral"]







def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()
    # allow   command line    to catch argument..

    return args


def main():
    # Argument parsing #################################################################
    c=0
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    print(args)

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) 


    face_cl= mp.solutions.face_mesh
    face= face_cl.FaceMesh()
 


    svm_model = joblib.load(svm_model_path)



    while True:
      

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
       

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        actualframe = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face.process(image)
        image.flags.writeable = True
        flg = False
        c=0 
        height, width, _ = image.shape
        thickness = 5
        ####################################### black frame ################
        frame = np.zeros((height, width, 3), dtype=np.uint8)

       
        cv.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 0), thickness)

        
       


        #  #############################################################################################################################
        if results.multi_face_landmarks is not None:
            for f in results.multi_face_landmarks:


               
                

               
                           
                brect = calc_bounding_rect(frame, f)
                landmark_point =[]
                
                for   i  in range (0, 468):
                    point = f.landmark[i]
                    x = min(int(point.x* width ), width - 1)
                    
                    y = min (int(point.y* height) , height-1)

        

                    landmark_point.append([x,y])
                    
                    cv.circle(frame ,(x,y),   2 ,  ( 100, 100, 0) , -1 )
           
            




            ### we get the image .. then we can proced with the following..
            
            

           
           
            offst=10
            cv.rectangle(frame, (brect[0]-offst, brect[1]-offst), (brect[2]+offst, brect[3]+offst),
                    (100, 100 ,100), 1)
            
            

            #nothing_landmarks = pre_process_landmark_nothing(landmark_point)
            #print(nothing_landmarks)
         
            pre_processed_landmark_list =pre_process_landmark(landmark_point)





############################## main code ###########
            face_index = svm_model.predict([pre_processed_landmark_list])[0]
            frame = draw_info_text( frame,
                                 brect,
                                       
                                   keypoint_classifier_labels[face_index])


            
            #print(len(landmark_point))
            #print (len(pre_processed_landmark_list))
     
            cv.imshow("image frame " , actualframe )
            cv.imshow('full frame ', frame)
          
    ################################################################################################################################
    





    cap.release()
    cv.destroyAllWindows()




def calc_bounding_rect(image, f):
    width, height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for   i  in range (0, 468):
        point = f.landmark[i]
        landmark_x = int(point.x* width )
        landmark_y = int(point.y* height)


        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point





    


def pre_process_landmark(landmark_list ):
    temp_landmark_list = copy.deepcopy(landmark_list)
   


    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        
        if index == 0:
           
            base_x, base_y = landmark_point[0], landmark_point[1]
            

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    #draw_face_landmarks(img, landmark_list)



   
  
   
    # Convert to a one-dimensional list

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    #print(temp_landmark_list)
    
    

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
 

    return temp_landmark_list



def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, hand_sign_text):
    cv.rectangle(image, (brect[0]-10, brect[1]-10), (brect[2]+10, brect[1]+10 - 46),
                 (100, 100, 0), -1)

    info_text = "Mood"
    if hand_sign_text != "":
        info_text = info_text + ':' + str(hand_sign_text)
        cv.putText(image, info_text, (brect[0]-10 + 5, brect[1]-10 - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
        

    # Draw "Hand Gesture" text at the top-left corner with double boundary

    return image





if __name__ == '__main__':
    c=0
    main()
