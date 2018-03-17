from Frames import Frames
from FlowGraph import FlowGraph
import numpy as np
import matplotlib.pyplot as plt

# load dummy video
video = Frames(dir_path= '../Data/VOT2016_GT/ball1',  size ='small', max_frames = 5)
# create a flow graph that matches the video
FG = FlowGraph(video.n_frames, video.dims, window = 7)

# number of edges (without those to the sink)
num_edges = len(list(FG.g.edges())) - video.dims[0]*video.dims[1]

#set edges to random costs
costs = np.random.random(num_edges)
FG.set_costs(costs)


# run update again to see whether there is improvement
for i in range(15):
    FG.update_costs(video.frames[0], video.frames[-1])

# get new costs
new_costs = FG.get_costs()

# check whether costs have changed
print(np.all(costs == new_costs))


# compute prediction and overlap
overlap = FG.compute_overlap(video.frames[0],video.frames[-1])
prediction = FG.prediction
print('Overlap is {0:.3f}.'.format(overlap))

x = np.sum(np.minimum(prediction, video.frames[-1]))
y = np.sum(np.maximum(prediction, video.frames[-1]))
print(x/y)
# plot prediction 

plt.figure()
plt.imshow(prediction)
plt.figure()
plt.imshow(video.frames[0])
plt.figure()
plt.imshow(video.frames[-1])

plt.show()

