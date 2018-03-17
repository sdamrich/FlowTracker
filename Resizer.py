import numpy as np
from Frames import Frames
from skimage.transform import resize
import os

dir_path= '../Data/VOT2016_GT/'
category = 'ball1'
GT = Frames(dir_path= dir_path + category, size = 'cropped')
GT_small = []
for frame in GT.frames:
    GT_small.append(np.round(resize(frame, (64, 64))))
    print('resized frame')
GT= np.array(GT_small)

np.save(os.path.join(dir_path+category,'small.npy'), GT_small)


video_path='../Data/vot2016/vot2016/'
video = Frames(dir_path=video_path+category, size ='cropped')

video_small = []
for frame in video.frames:
    video_small.append(resize(frame, (64, 64,3)))
    print('resized frame')
    
video_small= np.array(video_small)
np.save(os.path.join(video_path+category,'small.npy'), video_small)
