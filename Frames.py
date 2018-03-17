#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:06:19 2018

@author: sebuntu
"""

import numpy as np 
import os
import matplotlib.pyplot as plt

# Class for videos as frames
class Frames(object):
    def __init__(self, dir_path = None, video = None, size=None, max_frames = 50):
        self.max_frames = max_frames
        self.size = size
        # if a directory path is given, this overrides the video
        if video != None and dir_path != None:
            print("'video' must not be used if 'dir_path' is not None")
            print(dir_path + ' is used as directory path')
        
        if video != None and dir_path == None:
            cwd = os.getcwd()
            vot_dir = os.path.join(cwd, '../Data/vot2016/vot2016/')
            dir_path = os.path.join(vot_dir, video)
        
        if video == None and dir_path == None:
            print('Please specify a directory path or a video.')
        
        self.dir_path = dir_path
        # load the video
        self.load()
                
        
    # loads all .png or .jpg files in self.dir_path and stores them 
    # together with the number and dimension of the frames 
    def load(self):        
        self.frames = []
        
        if self.size==None:
            counter = 0
            for filename in os.listdir(self.dir_path):
                if (filename.endswith(".png") or filename.endswith('.jpg')) and counter < self.max_frames: 
                    self.frames.append(plt.imread(self.dir_path+'/'+filename))
                    counter += 1
                else:
                    continue
        elif self.size == 'cropped':
            cropped=np.load(self.dir_path + '/cropped.npy')
            self.frames= list(cropped)
        elif self.size == 'small':
            small=np.load(self.dir_path + '/small.npy')
            self.frames= list(small)
        else:
            print('Not a valid size.')
        
        self.frames = np.array(self.frames)
        self.frames = self.frames[0 : self.max_frames]            
        self.dims = self.frames[0].shape
        self.n_frames = len(self.frames)
    
    



