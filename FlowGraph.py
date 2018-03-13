from manage_ID import pixel2id, id2pixel
from graph_tool.all import *
import math
import numpy as np


class FlowGraph(object):
    # Constructor
    def __init__(self, num_frames, shape_frame, window): 
        # remember arguments
        self.num_frames = num_frames
        self.shape_frame = shape_frame
        
        # window size must be odd
        assert window % 2 == 1
        self.window = window
        
        self.pixel_per_frame = self.shape_frame[0] * self.shape_frame[1]
        self.num_pixels =  self.pixel_per_frame * self.num_frames + 1 
                   
        # build the graph given the metadata of the video
        self.build()
        
        # create edge property for edge costs
        eprop = self.g.new_edge_property("double")
        self.g.ep.cost = eprop
    
    # builds the graph
    def build(self):
        
        # instanciate a graph with all but the last node
        self.g = Graph()
        self.vertices = self.g.add_vertex(self.num_pixels - 1)
        
        # add edges from all vertices to the ones in the next frame
        for v in self.g.vertices():
            # is vertext not on last frame
            if int(v)+self.shape_frame[0]*self.shape_frame[1] <self.num_pixels - 1:
                frame_num, x ,y = id2pixel(int(v), self.shape_frame)
                self.add_edges2pixel(frame_num, x, y)
            # print status, once a frame is connected
            if int(v) % (self.shape_frame[0]*self.shape_frame[1]) == 0:
                print('Frame ' + str(int(v)//(self.pixel_per_frame)) + ' connected.')
                
        # insert sink with edges to it
        self.t = self.g.add_vertex()
        for x in range(self.shape_frame[0]):
            for y in range(self.shape_frame[1]):
                # add to sink
                #e = self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, self.num_frames -1, x, y)), self.t)
                # add edges from sink
                e = self.g.add_edge(self.t, self.g.vertex(pixel2id(self.shape_frame, self.num_frames-1, x, y)))

        
    # save and load methods. To be implemented
    def save(self):
        pass
    
    def load(self):
        pass

    # adds all edges to pixel of graph
    def add_edges2pixel(self, frame_num, x, y):
        # bounds centered window around pixel
        lower = int( - math.floor(self.window / 2))
        upper = int(math.floor(self.window / 2))
        # iterated through the neighbours in next frame and add edges \
        # provided neighbour does exist
        for i in range(lower, upper+1, 1 ):
            for j in range(lower , upper+1, 1 ):
                if x + i >= 0 and x + i < self.shape_frame[0]:
                    if y + j >= 0 and y +j < self.shape_frame[1]:
                        # add from earlier to later pixel
                        # self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, frame_num, x , y)), self.g.vertex(pixel2id(self.shape_frame, frame_num+1, x +i, y + j)))
                        # add from later to earlier pixel
                        self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, frame_num + 1, x , y)), self.g.vertex(pixel2id(self.shape_frame, frame_num, x +i, y + j)))
                        
    
    # sets costs of edges to passed array
    def set_costs(self, costs):
        # create edge property for edge costs
        eprop = self.g.new_edge_property("double")
        self.g.ep.cost = eprop
        # costs need to be concatenated with zeros for edges from sink to last frame
        self.g.ep.cost.a = np.hstack((costs, np.zeros(self.pixel_per_frame)))
    
    # returns current edge costs    
    def get_costs(self):
        return self.g.ep.cost.a[:-self.pixel_per_frame] # maybe need some +-1 here
        

    # updates the costs
    def update_costs(self, mask_f, mask_l):
        
        # run bellman ford (since there might be negative costs):
        success, dist_map, pred_map = bellman_ford_search(self.g, self.t, self.g.ep.cost)
        print('BF ran through without problems: '+ str(success))
        
        loss = 0
        hit = 0
        not_hit = 0
        # iterate over entire frame
        for x in range(len(mask_f)):
            for y in range(len(mask_f[0,:])):
                # only paths from pixels in BB matter
                if np.all(mask_f[x,y] == 1):
                    
                    # vertex that iterates along the path
                    v = self.g.vertex(pixel2id(self.shape_frame, 0, x,y))

                    # while we have not reached the first node (t), go to 
                    # predecessor and remember the edges we passed
                    path = []
                    while pred_map[v] != self.g.num_vertices()-1:
                        path.append(self.g.edge(pred_map[v], v))
                        v = pred_map[v]
                    
                    # obatin coodinates to pixel on last frame
                    _, v_x, v_y = id2pixel(int(v), self.shape_frame)
                    
                    # depedning on whether path lands in BB or not, 
                    # update edge costs on path and loss
                    length =len(path)
                    if np.all(mask_l[v_x, v_y] == 1):
                        f = 0 # distance from 
                        hit += 1
                        loss -= 1
                        for e in path:
                            f += 1  # increase factor with which we update the costs of an edge
                            self.g.ep.cost[e] -= 2*f/length
                    else:
                        f=0
                        not_hit +=1
                        loss +=1
                        for e in path:
                            f +=1
                            self.g.ep.cost[e] += 2*f/length
    
        # print some information                    
        print('Loss: '+ str(loss))
        print(str(hit) + ' of ' + str(hit+not_hit) + ' units of flow reached bounding box in last frame.') 
    
    
    
                    
    





