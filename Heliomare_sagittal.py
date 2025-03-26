# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:43:11 2023

@author: sdd380
"""

# Importing mediapipe and opencv 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Specify the video filename
path = 'E:/Treadmill/VIDEO DATA FOR SINA_ASKE/Video/'

folders = os.listdir(path)


# Loop over all participant folders and extract the video file names
for subject in folders:
    files = os.listdir(path+subject+'/video-sag/')
    files = list(filter(lambda f: f.endswith('.MOV'),files))
    subject = 'P19'
    for filename in files:
# # Loading in the video
        savename = os.path.splitext(filename)[0]
        num=""
        for i in savename:
            if i.isdigit():
                num = num+i  
        savename = subject + 'gait' + num[2:]        
        
        cap = cv2.VideoCapture(path + subject + '/video-sag/' + filename)
    
        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video file")
        
        
        left_anklex=[]
        left_ankley=[]
        left_anklez=[]
        
        right_anklex=[]
        right_ankley=[]
        right_anklez=[]
        
        left_heelx=[]
        left_heely=[]
        left_heelz=[]
        
        left_toex=[]
        left_toey=[]
        left_toez=[]
        
        right_heelx=[]
        right_heely=[]
        right_heelz=[]
        
        right_toex=[]
        right_toey=[]
        right_toez=[]
        
        left_kneex =[]
        left_kneey =[]
        left_kneez =[]
        
        right_kneex =[]
        right_kneey =[]
        right_kneez =[]
        
        left_hipx=[]
        left_hipy=[]
        left_hipz=[]
        
        right_hipx=[]
        right_hipy=[]
        right_hipz=[]
        
        
        
        # Read until video is completed
        with mp_pose.Pose(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as pose:
            while(cap.isOpened()):
              
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    # Create the pose estimation for this frame
                    # Recolor image to RGB
                    image = image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_hight, image_width, _ = image.shape
                    image.flags.writeable = True
                    
                       
                # Make detection
                    results = pose.process(image)
                    try:
                        landmarks= results.pose_landmarks.landmark
                    except:
                        pass
                    
                    # print(results.pose_landmarks)
                # Color back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
                # Drawing the landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS)
                    # Show the video
                    cv2.imshow('Mediapipe Feed', image)
                  
                    
                    left_anklex.append(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
                    left_ankley.append(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                    left_anklez.append(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
                    right_anklex.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
                    right_ankley.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                    right_anklez.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z)
        
                    left_heelx.append(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x)
                    left_heely.append(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y)
                    left_heelz.append(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z)
        
                    left_toex.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x)
                    left_toey.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)
                    left_toez.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z)
        
                    right_heelx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x)
                    right_heely.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y)
                    right_heelz.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z)
        
                    right_toex.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x)
                    right_toey.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)
                    right_toez.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z)
        
                    left_kneex.append(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x)
                    left_kneey.append(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                    left_kneez.append(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z)
        
                    right_kneex.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
                    right_kneey.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                    right_kneez.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z)
        
                    left_hipx.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x)
                    left_hipy.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                    left_hipz.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z)
        
                    right_hipx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
                    right_hipy.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                    right_hipz.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z)
        
            
        
        
                    # Press Q on keyboard to exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
          
        # Break the loop
                else:
                    break
        # When everything done, release
        # the video capture object
        cap.release()
          
        # Closes all the frames
        cv2.destroyAllWindows()
        
        
        output=np.column_stack((left_anklex,left_ankley,left_anklez,left_heelx,left_heely,left_heelz,
                                left_hipx,left_hipy,left_hipz,left_kneex,left_kneey,left_kneez,left_toex,
                                left_toey,left_toez,right_anklex,right_ankley,right_anklez,right_heelx,
                                right_heely,right_heelz,right_hipx,right_hipy,right_hipz,right_kneex,
                                right_kneey,right_kneez,right_toex,right_toey,right_toez))
        
        np.savetxt(path + subject + '/video-sag/' + filename+'.txt',output, delimiter=',', header= "left_anklex,left_ankley,left_anklez,left_heelx,left_heely,left_heelz,left_hipx,left_hipy,left_hipz,left_kneex,left_kneey,left_kneez,left_toex,left_toey,left_toez,right_anklex,right_ankley,right_anklez,right_heelx,right_heely,right_heelz,right_hipx,right_hipy,right_hipz,right_kneex,right_kneey,right_kneez,right_toex,right_toey,right_toez")
        print(filename)
