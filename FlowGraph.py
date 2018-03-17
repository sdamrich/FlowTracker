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
        # 'real' pixels, because later we will also add dummy pixels
        self.num_real_pixels =  self.pixel_per_frame * self.num_frames + 1 
                   
        # build the graph given the metadata of the video
        self.build()
        
        # create edge property for edge costs
        eprop = self.g.new_edge_property("double")
        self.g.ep.cost = eprop
    
    # builds the graph
    def build(self):
        
        # instanciate a graph with all but the last node
        self.g = Graph()
        self.vertices = self.g.add_vertex(self.num_real_pixels - 1)
        
        # insert sink with edges to it
        self.t = self.g.add_vertex()
        for x in range(self.shape_frame[0]):
            for y in range(self.shape_frame[1]):
                # add to sink
                #e = self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, self.num_frames -1, x, y)), self.t)
                # add edges from sink
                e = self.g.add_edge(self.t, self.g.vertex(pixel2id(self.shape_frame, self.num_frames-1, x, y)))        
        
        # add edges from all vertices to the ones in the next frame
        for v in self.g.vertices():
            # is vertext not on last frame
            if int(v)+self.shape_frame[0]*self.shape_frame[1] <self.num_real_pixels - 1:
                frame_num, x ,y = id2pixel(int(v), self.shape_frame)
                self.add_edges2pixel(frame_num, x, y)
            # print status, once a frame is connected
            if int(v) % (self.shape_frame[0]*self.shape_frame[1]) == 0 and int(v) >0:
                print('Frame ' + str(int(v)//(self.pixel_per_frame)) + ' connected.')
                


        
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
        # iterated through the neighbours in next frame and add edges 
        # provided neighbour does exist, if not create dummy neighbour and edge
        for j in range(lower, upper+1, 1 ):
            for i in range(lower , upper+1, 1 ):
                if x + i >= 0 and x + i < self.shape_frame[0] and y + j >= 0 and y +j < self.shape_frame[1]:
                    # add from earlier to later pixel
                    # self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, frame_num, x , y)), self.g.vertex(pixel2id(self.shape_frame, frame_num+1, x +i, y + j)))
                    # add from later to earlier pixel
                    self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, frame_num + 1, x , y)), self.g.vertex(pixel2id(self.shape_frame, frame_num, x +i, y + j)))
                else:
                    # add dummy node to graph and connect it to pixel, so that there are always window**2 many edges from every 'real' pixel
                    # the dummy pixels will have and index higher than the one of the sink
                    # the resulting graph with have more nodes and more edges than expected.
                    self.g.add_edge(self.g.vertex(pixel2id(self.shape_frame, frame_num + 1, x, y)), self.g.add_vertex())
                        
    
    # sets costs of edges to passed array
    def set_costs(self, costs):
        # create edge property for edge costs
        eprop = self.g.new_edge_property("double")
        self.g.ep.cost = eprop
        # costs need to be concatenated with zeros for edges from sink to last frame
        self.g.ep.cost.a = np.hstack((costs, np.zeros(self.pixel_per_frame)))
        
        # once new costs are given, compute flow
        self.compute_flow()
    
    # returns current edge costs    
    def get_costs(self):
        return self.g.ep.cost.a[:-self.pixel_per_frame] # maybe need some +-1 here
        
    
    # compute flow, returns predecessor map
    def compute_flow(self):
        # run bellman ford (since there might be negative costs):
        success, dist_map, pred_map = bellman_ford_search(self.g, self.t, self.g.ep.cost)
        print('BF ran through without problems: '+ str(success))
        self.pred_map = pred_map
        
        
    # generator for paths from boundingbox to sink
    def path_generator(self, mask_f):
        # iterate over entire frame
        for x in range(len(mask_f)):
            for y in range(len(mask_f[0,:])):
                # only paths from pixels in BB matter
                if mask_f[x,y] != 0:
                    # vertex that iterates along the path
                    v = self.g.vertex(pixel2id(self.shape_frame, 0, x,y))

                    # while we have not reached the first node (t), go to 
                    # predecessor and remember the edges we passed
                    path = []
                    while self.pred_map[v] != self.num_real_pixels-1:
                        path.append(self.g.edge(self.pred_map[v], v))
                        v = self.pred_map[v]
                    
                    # obatin coodinates to pixel on last frame
                    _, v_x, v_y = id2pixel(int(v), self.shape_frame)
                    yield path, v_x, v_y
    
    # computes a prediction of where the object is in the last frame
    # by the last nodes of the flow
    def predict(self, mask_f):
        paths = self.path_generator(mask_f)
        
        prediction = np.zeros(mask_f.shape)
        
        # set last pixels of paths to one
        for path, v_x, v_y in paths:
            prediction[v_x,v_y] = 1

        self.prediction = prediction
        return self.prediction
        
    # computes the overlap between the predicted and true position of 
    # the object in the last frame    
    def compute_overlap(self, mask_f, mask_l):
        self.predict(mask_f)
        self.overlap = np.sum(np.minimum(self.prediction, mask_l))/np.sum(np.maximum(self.prediction, mask_l)) 
        
        return self.overlap

    
    # updates the costs
    def update_costs(self, mask_f, mask_l):
        paths = self.path_generator(mask_f)
        for path, v_x, v_y in paths:  
            # depending on whether path lands in BB or not, 
            # update edge costs on path and loss
            length =len(path)
            if np.all(mask_l[v_x, v_y] == 1):
                f = 0 # distance from 
                for e in path:
                    f += 1  # increase factor with which we update the costs of an edge
                    self.g.ep.cost[e] -= 2*f/length
            else:
                f=0
                for e in path:
                    f +=1
                    self.g.ep.cost[e] += 2*f/length

    
    
                    
    





