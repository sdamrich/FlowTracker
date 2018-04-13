# FlowTracker
Using min-cost flows for tracking

This repository provides an implementation for and object tracking algorithm based on shortest paths for which the costs are given by a neural network. We used the Visual Object Tracking challenge dataset (and ground truth) for 2016 (available at http://www.votchallenge.net/vot2016/dataset.html). The datasets sould lie in the same directory as the FlowTracker directory for the paths to work. We first cropped the video "ball1" with Cropper.py. Then we resized it with Resizer.py. 

Frames.py defines a class for reading in the videos. NN.py defines the neural network that outputs the costs. FlowGraph.py defines the class for the graph in which we compute shortest paths. manage\_ID.py provides helper functions for addressing vertices of the graph. model.py links the neural network to the graph and defines the training and prediction methods of the model. The files starting with "test" implement small testing scripts. In particular, test\_model.py is the top level script where parameter such as iterations, window size etc can be set. For more information consult the report.
