# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 03:18:35 2018

@author: wangyi66
"""

import sys, os

raw_folder = sys.argv[1] #path that stores all raw txt, each of those store a 40 feature of one frame

for speaker in os.listdir(raw_folder):
    
    for root, dirs, files in os.walk(os.path.join(raw_folder, speaker)):
        
        for te_tr in dirs:
            new_path = root + '/' + te_tr + '_combined'
            try:
                os.mkdir(new_path)
            except:
                pass
        for file in files:
            seg_name = file.split('.')[0][:-8]+ '.txt'
            test_train = root.split('/')[-1]
            new_dir = test_train + '_combined'
            with open(os.path.join(root, file), 'r') as read:
                feature = read.readline()
            with open(os.path.join(root.rstrip(test_train), new_dir, seg_name), 'a') as write:
                write.write(feature + '\n')
            