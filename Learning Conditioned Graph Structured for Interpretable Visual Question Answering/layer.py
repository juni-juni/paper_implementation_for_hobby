import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

"""
Spatial Graph Convolution

* input & output
    
    input [batch_size, num_proposal, neighbor_size, input_feature]
    output [batch_size, num_proposal, neighbor_size, output_feature_size]

* Pipeline

    - Tensor size :
        + V : input, [batch, num_proposal, neighbor_size, input_feature]
        + P : polar_coordinate, [batch, num_proposal, neighbor_size, polar_coordinate_dim]  
        + G : Gaussian_Kernel_Weight, [batch*num_proposal, neighbor_size, num_kernel]
        + O : output of applying Gaussian kernel to input, [batch*num_proposal, num_kernel, input_feature]
        + H : concatenate the O [batch, output_feature]

    - Convolution
        + O = matmul(G_transpose, V)
        + H = [f_1(O_1),...,f_k(O_k)]

    - Gaussian_Kernel_Weight
        + 
          
"""


class Spatial_Graph_Convolution(nn.Module):

    def __init__(self, input_dim, output_dim, num_kernel):

        super(Spatial_Graph_Convolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernel = num_kernel

        #layers
        self.kernel_layers = nn.ModuleList([nn.Linear(input_dim, output_dim // num_kernel, bias=False) for i in range(num_kernel)])
        
        #Gaussian_kernel trainable parameters
        self.mean_length = Parameter(torch.Tensor(num_kernel, 1))
        self.sd_length = Parameter(torch.Tensor(num_kernel, 1))
        self.mean_angle = Parameter(torch.Tensor(num_kernel, 1))
        self.sd_length = Parameter(torch.Tensor(num_kernel, 1))


        
    
    def forward(self, in_feat, polar_coordinate):
        
        """
        in_feat - [batch * num_proposal, num_neighbor, in_feat_dim]
        polar_coordinate - [batch, num_proposal, num_neighbor, in_feat_dim]
        """
        
        num_neighbor = in_feat.size(1)

        gk_weights = self.get_gaussian_kernel_weight(polar_coordinate) # gk_weights - [batch * num_proposal * num_neighbor, num_kernel]
        gk_weights = gk_weights.view(-1, num_neighbor, self.num_kernel).transpose(1, 2) #gk_weights - [batch * num_proposal, num_kernel, num_neighbor]

        #applying gaussian weights
        kernel_output = torch.bmm(gk_weights, in_feat) # kernel_output - [batch * num_proposal, num_kernel, in_feat_dim]
        kernel_output = [self.kernel_layers[i](kernel_output[:,i]) for i in range(self.num_kernel)]
        output = torch.cat([k_output for k_output in kernel_output], dim=1)
        output = output.view(-1, self.output_dim)  # output - [batch * num_proposal, output_dim]

        return output




    def get_gaussian_kernel_weight(self, polar_coordinate):

        """
        polar_coordinate - [batch, num_proposal, num_neighbor, polar_coordinate_dim]
        weight - [batch * num_proposal * num_neighbor, num_kernel]
        """
        eps = 1e-14
        #Apply Gaussian kernel to length
        dist = (polar_coordinate[:,:,:,0].contiguous().view(-1, 1) - self.mean_length.view(1, -1))**2
        dist_weight = torch.exp(-0.5 * dist / (eps + self.sd_length.view(1, -1)**2))

        #Apply Gaussian Kernel to angle
        angle = torch.abs(polar_coordinate[:,:,:,1].contiguous().view(-1, 1) - self.mean_angle.view(1, -1))
        angle = torch.min(angle, 2 * np.pi - angle)
        angle_weight = torch.exp(-0.5 * (angle ** 2) / (eps + self.sd_length.view(1, -1))**2)

        #multiply distance weight and angle weight
        weight = dist_weight * angle_weight

        #filter out NaN values:
        weight[(weight != weight).detach()] = 0

        #Normalize
        weight = weight / torch.sum(weight, dim=1, keepdim=True)
        
        return weight
