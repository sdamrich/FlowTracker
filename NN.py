import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

# class for the network that computes the costs
class CostUNet(nn.Module):
    def __init__(self,num_frames, height, width, window):
        super(CostUNet, self).__init__()
        
        # remember padding and window size
        self.pad = math.floor((window-1)/2)
        self.window = window
        
        # first 2d conv layer
        self.conv_1 = nn.Conv3d(in_channels = 3, out_channels = 5, kernel_size = (1, window, window), padding = (0, self.pad, self.pad))
        # second 2d conv layer
        self.conv_2 = nn.Conv3d(in_channels = 5, out_channels = 10, kernel_size = (1, window, window), padding = (0, self.pad, self.pad))
        # pooling layer
        self.pool = nn.MaxPool3d(kernel_size = (1, 2, 2))
        # 3d conv layer with padding and depth kernel of 3
        self.conv_3 = nn.Conv3d(in_channels = 10, out_channels = 10, kernel_size = (3, window, window), padding = (1, self.pad, self.pad))
        # 3d conv layer without padding and depth kernel of 2
        self.conv_4 = nn.Conv3d(in_channels = 10, out_channels = 10, kernel_size = (2, window, window), padding = (0, self.pad, self.pad))
        # upsampling layer, mode can be nearest, trilinear 
        self.upsample = nn.Upsample(size = (num_frames-1, height, width), mode = 'trilinear')
        # transposed convolution layer for upsampling ( to double dims need stride = (1,2,2), pad = (0,x,x) and ker = (1, 2x+2, 2x+2)
        # self.upsample = nn.ConvTranspose3d(in_channels = 10, out_channels = 10, kernel_size = (1, 4, 4), padding = (0, 1, 1), stride = (1, 2,2))
        # 3d conv layer, that only convolutes in depth dimension
        self.conv_5 = nn.Conv3d(in_channels = 10, out_channels = 10,  kernel_size = (2, 1, 1))
        # conv layer to obtain edge weights
        self.conv_6 = nn.Conv3d(in_channels = 10, out_channels = window**2, kernel_size = (1, window, window), padding = (0, self.pad, self.pad))
        
        
    
    def forward(self, x):
        # reshape x to batchsize * channels * depth * height *width, usually channels come as last dimension...
        x = x.contiguous().view(x.size()[0], x.size()[-1], x.size()[1], x.size()[2], x.size()[3])
        # send input through 2d conv layers with Relus
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        # pool and send through 3d conv layers 
        y = self.pool(x)
        y = F.relu(self.conv_3(y))
        y = F.relu(self.conv_4(y))
        # upsample
        y = self.upsample(y)
        # send x throuh depth convolution layer
        x = F.relu(self.conv_5(x))
        
        # add x and y and send through linear layer
        z = x + y
        c = F.relu(self.conv_6(z))
        
        print('Forward pass done.')
        
        return c
        

        
    


