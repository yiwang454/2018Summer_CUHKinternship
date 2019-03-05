# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 05:41:37 2018

@author: wangyi66
"""

from statistic_nem import video_count
import os

video_list = ['F02','F04','F05','M08','M14','M16']
path = os.getcwd()

# a dictionary that use name of words as key to record for that word how many speakers failed (30%)
word_dic = {} 
word_dic_half = {} # here: 50%

#a list contain function return for each name
word_list = []

for name in video_list:
    word_dic, total_word, bad_dic_3, bad_dic_5, number_tuple = video_count(name, path)
    word_list.append((bad_dic_3, bad_dic_5))
    
for i in range(len(word_list)):
    for items in word_list[i][0]:
        if items[7:] in word_dic:
            word_dic[items[7:]] += 1
        else:
            word_dic[items[7:]] = 1

print(word_dic)
         