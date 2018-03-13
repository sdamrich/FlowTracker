import numpy as np
from Frames import Frames
dir_path= '/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/'
category = 'ball1'
frames = Frames(dir_path=dir_path+category)

min_y = frames.dims[1]
min_x = frames.dims[0]
max_x = 0
max_y = 0

for frame in frames.frames:
    print(frame.shape)
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            if frame[x,y] == 1:
                if min_y > y:
                    min_y = y
                if max_y < y:
                    max_y = y
                if min_x > x: 
                    min_x = x
                if max_x < x:
                    max_x = x

min_y +=10
max_y -=10
min_x +=10
max_x -=10

print(min_x, max_x, min_y, max_y)
cropped_frames = []
for frame in frames.frames:
    frame = frame[min_x:max_x, min_y:max_y]
    cropped_frames.append(frame)
    
cropped_frames=np.array(cropped_frames)
np.save(dir_path + category +'/cropped.npy', cropped_frames)
shapes=np.array([min_x, max_x, min_y, max_y])
np.save(dir_path + category + '/shape.npy', shapes)

video_path='/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/vot2016/vot2016/'
video = Frames(dir_path=video_path+category)

cropped_video = []
for frame in video.frames:
    frame = frame[min_x:max_x, min_y :max_y]
    cropped_video.append(frame)
    
cropped_video = np.array(cropped_video)
np.save(video_path+category + '/cropped.npy', cropped_video)


