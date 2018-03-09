from manage_ID import pixel2id, id2pixel
from graph_tool.all import *
from Frames import Frames
import math

# adds all edges to pixel of given graph
def add_edges2pixel(graph, shape_frame, frame_num, x, y, window):
    # window size must be odd
    assert window % 2 == 1
    lower = int( - math.floor(window / 2))
    upper = int(math.ceil(window / 2))
    for i in range(lower, upper+1, 1 ):
        for j in range(lower , upper+1, 1 ):
            if x + i >= 0 and x + i < shape_frame[0]:
                if y + j >= 0 and y +j < shape_frame[1]:
                    #print(frame_num, x+ i , y + j )
                    #print(pixel2id(shape_frame, frame_num + 1, x + i , y + j))
                    graph.add_edge(graph.vertex(pixel2id(shape_frame, frame_num, x , y)), graph.vertex(pixel2id(shape_frame, frame_num+1, x +i, y + j)))
                

frames = Frames(dir_path= '/media/sebuntu/Daten/Daten/Studium/Semester_13/ML4CV/project/Data/VOT2016_GT/ball1test')

window = 3

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

for v in g.vertices():
    if int(v)+frames.dims[0]*frames.dims[1] <n_pixels:
        frame_num, x ,y = id2pixel(int(v), frames.dims)
        add_edges2pixel(g, frames.dims, frame_num, x, y , window)
    if int(v) % (frames.dims[0]*frames.dims[1]) == 0:
        print('Frame ' + str(int(v)//(frames.dims[0]*frames.dims[1])) + ' connected.')
        
#graph_draw(g)
        

for v in g.vertex(pixel2id(frames.dims,2, 0,0)).out_neighbors():
    print(id2pixel(int(v), frames.dims))
    
eprop = g.new_edge_property("double")
g.ep.cost = vprop                        # equivalent to g.vertex_properties["foo"] = vprop
v = g.vertex(0)


