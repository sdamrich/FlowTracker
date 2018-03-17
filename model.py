import numpy as np
from Frames import Frames
from FlowGraph import FlowGraph
from NN import CostUNet
import torch
from torch.autograd import Variable

# class for whole model
class FlowModel(object):
    # Constructor
    def __init__(self, num_frames, height, width, window):
        # remember parameters
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.window = window
        # create FlowGraph
        self.FG = FlowGraph(num_frames, (height, width), window)
        # compute how many edges there are (including dummy edges but discarding those to the sink)
        self.edges_per_frame = width * height * window**2
        # create Neural Network
        self.net = CostUNet(num_frames, height, width, window)
        
        # build create loss function and optimiser
        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        
        
    # trains the neural network    
    def train(self, videos, GT, iterations):
        num_videos = len(videos)
        # set up input variable from videos
        inp = Variable(torch.from_numpy(videos).float(), requires_grad = False)
        
        for i in range(iterations):
            # compute costs from NN and reshape them to num_videos * edges per video
            costs_torch = self.net(inp)
            
            # convert costs to numpy
            costs = costs_torch.data.numpy() 
            # costs has shape num_videos *channels* num_frames -1 * height * width
            # the edges in the graph are created to be result of reshape on 
            # num_videos * num_frames -1 * height * width * channels
            # so we move axis of channels accordingly
            costs = np.moveaxis(costs, 1, -1)
            
            # interchange direction of edges per pixel
            #costs = costs.reshape(num_videos, self.num_frames-1, self.height, self.width, self.window, self.window)
            #costs = np.moveaxis(costs, -1, -2)
            #costs = costs.reshape(num_videos, self.num_frames-1, self.height, self.width, self.window**2)

            
            # and then reshape 
            costs = costs.reshape(num_videos, -1)
            
            
            # Placeholder for new costs
            new_costs = np.zeros(costs.shape)
            for j in range(num_videos):
                # for every video in trainings batch, set costs of graph to the costs computed for that video
                self.FG.set_costs(costs[j])
                # update the costs
                self.FG.update_costs(GT[j][0], GT[j][1])
                # obtain the new costs for the video
                new_costs[j] = self.FG.get_costs()
             
            # reshape new costs back into format of costs
            new_costs = new_costs.reshape((num_videos, self.num_frames-1, self.height, self.width, self.window**2))

            # interchange direction of edges per pixel
            #new_costs = new_costs.reshape(num_videos, self.num_frames-1, self.height, self.width, self.window, self.window)
            #new_costs = np.moveaxis(new_costs, -1, -2)
            #new_costs = new_costs.reshape(num_videos, self.num_frames-1, self.height, self.width, self.window**2)


            # move axis back to correct position
            new_costs = np.moveaxis(new_costs, -1, 1)
            
            # give values back to torch variable
            new_costs_torch = Variable(torch.from_numpy(new_costs).type(torch.FloatTensor), requires_grad = False)

            # compute loss and print it                
            self.loss = self.loss_fn(costs_torch, new_costs_torch)
            if i % (iterations/10) == 0:
                print('Iteration: {}, Loss for cost: {}'.format(i, self.loss.data[0]))
            
            # reset gradients, compute new gradients and update parameters
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
            
    # predicts the postition of object in last frame for one video
    # if GT for last frame is given, also computes how much prediction 
    # and groud truth overlap (0 complete overlap, 1 no overlap at all)
    def predict(self, video, mask_f, mask_l = None):
        # add dummy dimension
        video = np.expand_dims(video, axis = 0)
        
        # set up input variable
        inp = Variable(torch.from_numpy(video).float(), requires_grad = False)
        # compute costs
        costs_torch = self.net(inp)
        
        # convert costs to numpy
        costs = costs_torch.data.numpy() 
        # costs has shape num_videos *channels* num_frames -1 * height * width
        # the edges in the graph are created to be result of reshape on 
        # num_videos * num_frames -1 * height * width * channels
        # so we move axis accordingly
        costs = np.moveaxis(costs, 1, -1)
        # and then reshape 
        costs = costs.reshape(1, -1)
        
        # feed costs to FlowGraph
        self.FG.set_costs(costs[0])
        
        # predict or comput overlap and predict
        if np.any(mask_l == None):
            self.prediction = self.FG.predict(mask_f)
        else:
            self.overlap = self.FG.compute_overlap(mask_f, mask_l)
            self.prediction = self.FG.prediction
            print('Overlap of prediction of ground truth is {0:.3f}.'.format(self.overlap))
        
        
        
            
            
            
