from manage_ID import pixel2id, id2pixel
from graph_tool.all import *
from Frames import Frames
import math
import numpy as np

# adds all edges to pixel of given graph
def add_edges2pixel(graph, shape_frame, frame_num, x, y, window):
    # window size must be odd
    assert window % 2 == 1
    lower = int( - math.floor(window / 2))
    upper = int(math.floor(window / 2))
    for i in range(lower, upper+1, 1 ):
        for j in range(lower , upper+1, 1 ):
            if x + i >= 0 and x + i < shape_frame[0]:
                if y + j >= 0 and y +j < shape_frame[1]:
                    # add from earlier to later pixel
                    # graph.add_edge(graph.vertex(pixel2id(shape_frame, frame_num, x , y)), graph.vertex(pixel2id(shape_frame, frame_num+1, x +i, y + j)))
                    # add from later to earlier pixel
                    graph.add_edge(graph.vertex(pixel2id(shape_frame, frame_num + 1, x , y)), graph.vertex(pixel2id(shape_frame, frame_num, x +i, y + j)))

def update_costs(g, pred_map, frame_0, frame_l):
    loss = 0
    hit = 0
    not_hit = 0
    # iterate over entire frame
    for x in range(len(frame_0)):
        for y in range(len(frame_0[0,:])):
            # only paths from pixels in BB matter
            if np.all(frame_0[x,y] == 1):
                
                # vertex that iterates along the path
                v = g.vertex(pixel2id(frame_0.shape, 0, x,y))

                # while we have not reached the first node (t), go to 
                # predecessor and remember the edges we passed
                path = []
                while pred_map[v] != g.num_vertices()-1:
                    path.append(g.edge(pred_map[v], v))
                    v = pred_map[v]
                
                # obatin coodinates to pixel on last frame
                _, v_x, v_y = id2pixel(int(v), frame_l.shape)
                
                # depedning on whether path lands in BB or not, 
                # update edge costs on path and loss
                length =len(path)
                if np.all(frame_l[v_x, v_y] == 1):
                    f = 0
                    hit += 1
                    loss -= 1
                    for e in path:
                        f += 1
                        factor= f/length
                        g.ep.cost[e] -= 2*f/length
                else:
                    f=0
                    not_hit +=1
                    loss +=1
                    for e in path:
                        f +=1
                        g.ep.cost[e] += 2*f/length
    return loss, hit, not_hit 
    
    
    
def learn_costs(g, frame_0, frame_l, iterations):
    for i in range(iterations):
        # run bellman ford (since there might be negative costs):
        success, dist_map, pred_map = bellman_ford_search(g, t, g.ep.cost)
        print('BF ran through without problems: '+ str(success))
        
        print('Loss: '+ str(update_costs(g, pred_map, frame_0, frame_l)))
        print('Iteration %d done.' % i)

    
                    
                    
                    
    

frames = Frames(dir_path= '/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/ball1', npy = True)
print('Number of pixels in source BB :' , np.sum(frames.frames[0]))
print('Number of pixels in target BB :' , np.sum(frames.frames[frames.n_frames-1]))
window =5

n_pixels = frames.dims[0]*frames.dims[1] * frames.n_frames

g = Graph()

vertices = g.add_vertex(n_pixels)


"""
for frame_num in range(frames.n_frames-1):
    print('Edges to frame ' +str(frame_num)+ ' are being added')
    for x in range(frames.dims[0]):
        for y in range(frames.dims[1]):
            add_edges2pixel(g, frames.dims, frame_num, x , y , window)
"""

# for all vertices, add edges to next frame
for v in g.vertices():
    if int(v)+frames.dims[0]*frames.dims[1] <n_pixels:
        frame_num, x ,y = id2pixel(int(v), frames.dims)
        add_edges2pixel(g, frames.dims, frame_num, x, y , window)
    if int(v) % (frames.dims[0]*frames.dims[1]) == 0:
        print('Frame ' + str(int(v)//(frames.dims[0]*frames.dims[1])) + ' connected.')
        
#graph_draw(g)
        

# print out neighbours for a test pixel
#for v in g.vertex(pixel2id(frames.dims,1, 1,2)).out_neighbors():
#<    print(id2pixel(int(v), frames.dims))

# create edge property for edge costs
eprop = g.new_edge_property("double")
g.ep.cost = eprop                        # equivalent to g.vertex_properties["foo"] = vprop

# randomly set edge costs
g.ep.cost.a = 2* np.random.random(g.num_edges())-1

# print some edge costs as test
#for e in g.vertex(pixel2id(frames.dims,1, 1,2)).out_edges():
#    print(g.ep.cost[e])
    

# insert sink with edges to it and add set edge costs to 0
t = g.add_vertex()
for x in range(frames.dims[0]):
    for y in range(frames.dims[1]):
        # add to sink
        #e = g.add_edge(g.vertex(pixel2id(frames.dims, frames.n_frames-1, x, y)), t)
        # add edges from sink
        e = g.add_edge(t, g.vertex(pixel2id(frames.dims, frames.n_frames-1, x, y)))
        g.ep.cost[e] = 0


learn_costs(g, frames.frames[0], frames.frames[frames.n_frames-1], 100)







