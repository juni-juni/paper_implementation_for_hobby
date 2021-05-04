import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, emb_channels=[3, 64, 64*2, 64*4, 64*8], num_classes=10, ):
        super(Discriminator, self).__init__()

        nn_list = []
        for i in range(1, len(emb_channels)):
            in_dim = channels[i-1]
            out_dim = channels[i]
            nn_list.append(nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1, bias=True))
            nn_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            nn_list.append(nn.BatchNorm2d(num_features=out_dim))
        
        self.embedding = nn.Sequential(*nn_list)
        self.real_fake_classifier = nn
