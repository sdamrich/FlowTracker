import numpy as np
import matplotlib.pyplot as plt
from Frames import Frames
frames = np.load('/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/ball1/cropped.npy')
video = np.load('/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/vot2016/vot2016/ball1/cropped.npy')

plt.figure(1)
plt.imshow(frames[0, : , :])

plt.figure(2)
plt.imshow(video[0, :,:])
plt.show()
frames = Frames(dir_path = '/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/ball1', npy=True)

print(len(frames.frames))
