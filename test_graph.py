from Frames import Frames
from FlowGraph import FlowGraph
import numpy as np

# load dummy video
video = Frames(dir_path= '/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/ball1test')

# create a flow graph that matches the video
FG = FlowGraph(video.n_frames, video.dims, window = 5)

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
