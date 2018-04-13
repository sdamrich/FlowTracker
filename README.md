# FlowTracker
Using min-cost flows for tracking

This repository provides an implementation of an object tracking algorithm based on shortest paths for which the costs are given by a neural network. We used the Visual Object Tracking challenge 2016 dataset and ground truth (available at http://www.votchallenge.net/vot2016/dataset.html). The datasets sould lie in the same directory as the FlowTracker directory so that the relative file paths work.

## How to use the repository
Frames.py defines a class for reading in the videos. Cropper.py crops a video to an area around the object of interest. Resizer.py then resizes it to the desired dimensions. NN.py defines the neural network that outputs the costs. FlowGraph.py defines the graph in which we compute shortest paths. manage\_ID.py provides helper functions for addressing vertices of the graph. model.py links the neural network to the graph and defines the training and prediction methods of the model. The files starting with "test" implement small testing scripts. In particular, test\_model.py is the top level script where parameters such as iterations, window size etc can be set. Running this script trains the model (on the preprocessed video "ball1") and afterwards predicts the object's location in the last frame. For more information please consult the report.
