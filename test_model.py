from model import FlowModel
from Frames import Frames
import matplotlib.pyplot as plt
import numpy as np

# hyperparameter
max_frames = 5
window = 5
iterations = 10



# load GT and video
GT = Frames(dir_path= '..//Data/VOT2016_GT/ball1', size ='small', max_frames = max_frames)
video = Frames(dir_path= '..//Data/vot2016/vot2016/ball1', size ='small', max_frames = max_frames)


# create a flow model
FM = FlowModel(video.n_frames, video.dims[0], video.dims[1], window)


# train the model, need to expand dims as model expects batch (here of size one)
FM.train(np.expand_dims(video.frames, axis = 0), np.expand_dims(GT.frames, axis =0), iterations)

# predict the position of the object in the last frame
FM.predict(video.frames, GT.frames[0], GT.frames[1])

# print the GT for the first and last frame as well as the prediction
plt.figure()
plt.title('Ground Truth first Frame')
plt.imshow(GT.frames[0])

plt.figure()
plt.title('Prediction')
plt.imshow(FM.prediction)

plt.figure()
plt.title('Ground Truth last Frame')
plt.imshow(GT.frames[-1])

plt.show()




