#Supplemental Deatial information

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



"""The SDI method.
   Parameters: in_planes,
               out_planes,
			   shrink_factor: limit the parameters of predicting weight module
		
   Note:(1) self.share_conv is the shared convolution filters by the pure and noisy feature

        (2) self.norm_weight_miu,self.norm_weight_sigma are the parameters of Gaussian noise distribution
            self.uniform_weight is the parameter of Uniform distribution
		    
		    
			
			
   Example: self.Addi_Reg=Addi_Reg(planes,planes, shrink_factor=shrink_factor,stride=stride)
"""	
	
class SDI(nn.Module):
    def __init__(self, in_planes,out_planes, channels, height, width,shrink_factor=8,stride=1):
        super(SDI, self).__init__()

        self.embedded_vector=nn.Parameter(torch.cuda.FloatTensor(1,1,height,width).normal_(0,1)*0.1,requires_grad = True)
        self.project = nn.Conv2d(1, channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        
        mip = max(8, out_planes // shrink_factor)
        self.conv_p1 = nn.Conv2d(in_planes, mip, kernel_size=1, padding=0,bias=False)
        self.conv_p2 = nn.Conv2d(mip,out_planes, kernel_size=1, padding=0,bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        
        Spatial_info = self.project(self.embedded_vector) #size:(1,C,H,W)
        Channel_info = F.avg_pool2d(x,x.shape[-1])
        Channel_info = F.sigmoid(self.conv_p2(self.relu(self.conv_p1(Channel_info))))
        Info = Spatial_info * Channel_info
        
        return x+Info
