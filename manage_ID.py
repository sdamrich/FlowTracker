import numpy as np
import math

# computes the id of a pixel, given its frame, position in frame (x,y )
# and the shape of the frame
def pixel2id(shape_frame, frame_num, x, y ):
    assert x >= 0 and x < shape_frame[0]
    assert y >= 0 and y < shape_frame[1]
    return shape_frame[0]*shape_frame[1]*frame_num + y*shape_frame[0] + x
    
    
# retrieves the pixel from its id
def id2pixel(pixel_id, shape_frame):
    x = pixel_id % shape_frame[0]
    y = int( math.floor((pixel_id % (shape_frame[0]*shape_frame[1])) / shape_frame[0]) )
    frame_num = int( math.floor(pixel_id / (shape_frame[0]*shape_frame[1])) )
    return frame_num, x ,y 
