# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:27:06 2018

@author: mansi
"""
import numpy as np
def format_time(b):
    new_b = []
    new_time ='';
    for time in b:
        if(len(time)==1):
            new_time = "0" +time + "00"
            print(new_time)
        elif(len(time)==2):
            new_time = time + "00"
        elif(len(time)==3):
            new_time = "0" + time
        elif(len(time) == 4):
            new_time = time
        #np.append(new_b, np.array([new_time]))
        new_b.append(int(new_time[0:2]))
        
    return new_b
        
b=np.array(['1', '1740', '835'])
        
new_b = format_time(b)

print(new_b)
