# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 03:24:36 2018

@author: wangyi66
"""
import random, os
import numpy as np

list_info = [] #list of total frames number and deleted frames
add_auto_output = '.\\fake_data'#folder that saved the autoencoder output txt     to be complete
txt_frame_info = '.\\trial_data.txt'
dict_multiple = {} #a dictionary whose key is the video name and stores a list of number to guide duplication

file = open(txt_frame_info, 'r')

def turn_to_int(it):
    if it.isalnum():
        return int(it)
    else:
        return it

#read in frame numbers
count = 0
for line in file:
    if count !=0 :
        frame_list = line.strip('\n').split('\t')#not sure if this split works
        #list: 0: name 1: video_frame 2: audio_frame 4: deleted_frame
        frame_list = list(map(turn_to_int, frame_list))
        list_info.append(frame_list)
        #print(frame_list)
        #generate a list of numbers used to guide duplicating process & store in a dict
        rand_numbers = random.sample(range(frame_list[1]),frame_list[4])
        
        dict_multiple[frame_list[0]] = (rand_numbers, frame_list[2])
        #print(dict_multiple[frame_list[0]])
    count += 1
        
file.close()

#read in features
for root, dirs, txt in os.walk(add_auto_output):
    
    for file in txt:
        list_feature = []
        print(os.path.join(root, file))
        feature = np.loadtxt(os.path.join(root, file))
        name = file.strip('.txt')
        rand_numbers = dict_multiple[name][0]
        num = 0
        for line in feature:
            if num in rand_numbers:
                list_feature.append(line)
                list_feature.append(line)
                list_feature.append(line)
            else:
                list_feature.append(line)
                list_feature.append(line)
                list_feature.append(line)
                list_feature.append(line)
            num += 1
        
        print(len(list_feature))
        
        if len(list_feature) == dict_multiple[name][1]:
        
            final_out = open(name + '.feature40.auto.txt', 'a')
            sep = '\t'
            for member in list_feature:
                char_list = list(map(lambda x: str(x), member))
                out_str = sep.join(char_list)
                print(char_list)
                final_out.write(out_str + '\n')
                #print(member)
            final_out.close()
        
        else:
            print('error occur: ' + name)