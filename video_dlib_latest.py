# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 06:53:01 2018

@author: wangyi66
"""

import cv2
import sys
import dlib
import os
import numpy as np


path = sys.argv[1]
if path[-1] != '/':
    print('argv1 with wrong format')
    os._exit(0)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increase_contrast(img, contrast=64):
    
    img = np.int16(img)  
    img = img*(contrast/127 + 1) - contrast
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return  img


def illuminance_normalization(img):
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    return bgr
    
def detect_one_face(img, shape):
    
    list_of_x = []
    list_of_y = []    
    list_of_nose = []
    list_of_chin = []
    for index, pt in enumerate(shape.parts()):    
                    
        if index in list(range(5,11)):
            list_of_chin.append(pt.y)
            pt_pos = (pt.x, pt.y)
                        
        if index in list(range(31,36)):
            list_of_nose.append(pt.y)
            pt_pos = (pt.x, pt.y)
                    
        if index in list(range(48,68)):
            list_of_x.append(pt.x)
            list_of_y.append(pt.y)
            pt_pos = (pt.x, pt.y)
                            
        
    max_x = max(list_of_x)
    min_x = min(list_of_x)
    left_x = int(1.2*min_x - 0.2*max_x)
    right_x = int(1.2*max_x - 0.2*min_x)
    chin_y = max(list_of_y)
    nose_y = min(list_of_y)
    up_y = int(chin_y*0.25 + max(list_of_chin)*0.75)
    down_y = int(nose_y* 0.75 + max(list_of_nose)*0.25)
    retval = [left_x, right_x, down_y, up_y]
    
    return retval

def finding_face(img, img_process, img_error, img_name, error, detector, predictor):
    
    dets, scores, idx = detector.run(img_process, 1)
            
    if len(dets) <= 0 :
        
        r, g, b = cv2.split(img_process) 
        img_process = cv2.merge([b, g, r])
        error += 1
        return error, img_error
        
            
    else:
        scores_max = 0
        useful_face = 0
        for INDEX, face in enumerate(dets):
            if scores[INDEX] > scores_max:
                scores_max = scores[INDEX]
                useful_face = INDEX
                
        shape = predictor(img_process, dets[useful_face])  
        re = detect_one_face(img, shape)
        
    
        if (re[3] - re[2])>= 40 or (re[1] - re[0]) >= 60 :
        
            img_mouth = cv2.resize(img[re[2]:re[3],re[0]:re[1]], (128, 128), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img_mouth, cv2.COLOR_BGR2GRAY)
            return error, gray
        
        else:
            error += 1
            print('detect error: too small, probably wrong')
            return error, img_error
            
def main_process(video_name):                 

    current_path = os.getcwd()  
    
    vidcap = cv2.VideoCapture(path + video_name + '.mp4')                                 
    success,img = vidcap.read()
    img_error = np.zeros((128,128,3),dtype="uint8") 
    predictor_path = current_path + '/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor(predictor_path)

    os.mkdir(current_path + '/' + video_name)    
    
    count = 0
    error = 0
    img_list = []
    while (True): 
        img_name = current_path + '/' + video_name + '/' + video_name + '_%07d'% count + '.jpg'
    
        if success == True:
        
            b, g, r = cv2.split(img) 
            img2 = cv2.merge([r, g, b])
        
            error, img3 = finding_face(img, img2, img_error, img_name, error, detector, predictor)
            img_list.append((img_name, img3))
        
        else:
            print('detection end')
            break
    
        success,img = vidcap.read()
        count += 1

    error_rate = error / float(count)
    print(count , error, error_rate)

    if error_rate >= 0.3:
        count = 0
        error_m = 0
        img_list_m = []
        vidcap = cv2.VideoCapture(path + video_name + '.mp4')                                 
        success,img = vidcap.read()
        
        while (True): 
            
            img_name = current_path + '/' + video_name + '/' + video_name + '_%07d'% count + '.jpg'
            if success == True:
                
                img2 = illuminance_normalization(img)
                b, g, r = cv2.split(img2) 
                img3 = cv2.merge([r, g, b])
                
                error_m, img4 = finding_face(img, img3, img_error, img_name, error, detector, predictor)
                img_list_m.append((img_name, img4))
            else:
                print('detection end')
                break
                
            success,img = vidcap.read()
            count += 1
        
        print(count , error_m, error_m / float(count))
        
        if error > error_m:
            for image in img_list_m:
                cv2.imwrite(image[0], image[1])
        else:
            for image in img_list:
                cv2.imwrite(image[0], image[1])
    else:
        for image in img_list:
            cv2.imwrite(image[0], image[1])


for FILE in os.listdir(path):
    
    file_name = os.path.splitext(FILE)
    
    if file_name[1] == '.mp4':
        try:
            main_process(file_name[0])
        except:
            print('error occur for' + file_name[0] )


