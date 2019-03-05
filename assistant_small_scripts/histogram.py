# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 07:50:58 2018

@author: wangyi66
"""
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = 'E:\\photo\\train_video_lip' # the path of the directory containing all the word albums
#sys.argv[1]
#'C:\\Users\\wangyi66\\Desktop\\py\\M16_B1_CW1_M1'
histo_data = [] #to record the span of every single gap

for directory in os.listdir(current_path):
    
    zero_one = []
        
    for files_n in os.listdir(current_path + '\\' + directory):
        img = cv2.imread(current_path + '\\' +directory +'\\' + files_n, 0)
        
        all_value = cv2.minMaxLoc(img)
        MAX = all_value[1]
        if MAX == 0:
            zero_one.append(1)
        else:
            zero_one.append(0)
        
    print(zero_one)
    
    while(len(zero_one) != 0):
        if zero_one[0] == 0:
            zero_one = zero_one[1:]
                
        elif (0 in zero_one) == True:
            z_pos = zero_one.index(0)
            histo_data.append(z_pos)
            zero_one = zero_one[z_pos:]
        else:
            histo_data.append(len(zero_one))
            break
                
a = np.asarray(histo_data)
plt.hist(a, bins= list(range(min(histo_data), (max(histo_data)+1))))    
plt.title("Histogram of undetected seqeunces")
plt.savefig("Histogram.png")
            
