"""Build models"""
"""Module for building CNN"""

import torch
import torch.nn as nn
import numpy as np


class CNNordinalclassifier(nn.Module):
    def __init__(self, in_data1, in_data2, out_size):
        super(CNNordinalclassifier, self).__init__()

        # out_size = out_data.size(-1)
        self.inputshape1 = in_data1.size()
        self.inputshape2 = in_data2.size()
        self.n_filters = [32,32]
        self.kernel_size = [3,3,3]
        self.pool_width = [2,2,2]
        self.hiddens = [150,80,70]

        self.conv1 = self.convblock(self.inputshape1[1],self.n_filters[0],self.kernel_size[0],self.pool_width[0])

        self.conv2 = self.convblock(self.n_filters[0],self.n_filters[1],self.kernel_size[1],self.pool_width[1])

        self.conv3 = self.convblock(self.n_filters[1],self.n_filters[2],self.kernel_size[2],self.pool_width[2])

        self.flatten = nn.Flatten()
        flatdims = self.reduce_pool()
        flatsize = np.prod(flatdims)

        self.DenseLayer1 = nn.Linear(flatsize+self.inputshape2[1],self.hiddens[0])
        self.Sigmoid1 = nn.ReLU()

        self.DenseLayer2 = nn.Linear(self.hiddens[0],self.hiddens[1])
        self.Sigmoid2 = nn.ReLU()
        self.outlayer = nn.Linear(self.hiddens[1],out_size)
        self.sigmoidout = nn.Sigmoid()

    def convblock(self, in_channels, out_channels, kernel_size, pool_width=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.AvgPool2d(pool_width),
            nn.Sigmoid(),
        )
    
    def reduce_pool(self):
        # only works for square pooling
        # also only works with "same" padding
        dimreduce = []
        dimreduce.append([self.n_filters[0],self.inputshape1[2]//self.pool_width[0],self.inputshape1[3]//self.pool_width[0]])

        for i in range(len(self.pool_width)-1):
            dimreduce.append([self.n_filters[i+1],dimreduce[i][1]//self.pool_width[i+1],dimreduce[i][2]//self.pool_width[i+1]])

        return dimreduce[-1]

    def forward(self, x,x2):
        # Contracting path

        convblock1 = self.conv1(x)
        convblock2 = self.conv2(convblock1)
        bottleneck = self.conv3(convblock2)


        #flatten convolutions
        bottleneckflat = self.flatten(bottleneck)
        bottleneckandone = torch.cat([bottleneckflat,x2],dim=1)
        # print(bottleneckandone.size())

        # pass to dense layers
        Dense1 = self.DenseLayer1(bottleneckandone)
        Dense1 = self.Sigmoid1(Dense1)


        Dense2 = self.DenseLayer2(Dense1)
        Dense2 = self.Sigmoid2(Dense2)

        outfull = self.outlayer(Dense2)
        outsigmoid = self.sigmoidout(outfull)

        return outsigmoid


class CNNvanillaclassifier(nn.Module):
    def __init__(self, in_data1, in_data2, out_data):
        super(CNNvanillaclassifier, self).__init__()

        out_size = out_data.size(-1)
        self.inputshape1 = in_data1.size()
        self.inputshape2 = in_data2.size()
        self.n_filters = [16,16,16]
        self.kernel_size = [3,3,3]
        self.pool_width = [2,2,2]
        self.hiddens = [100,50]

        self.conv1 = self.convblock(self.inputshape1[1],self.n_filters[0],self.kernel_size[0],self.pool_width[0])

        self.conv2 = self.convblock(self.n_filters[0],self.n_filters[1],self.kernel_size[1],self.pool_width[1])

        self.conv3 = self.convblock(self.n_filters[1],self.n_filters[2],self.kernel_size[2],self.pool_width[2])

        self.flatten = nn.Flatten()
        flatdims = self.reduce_pool()
        flatsize = np.prod(flatdims)

        self.DenseLayer1 = nn.Linear(flatsize+self.inputshape2[1],self.hiddens[0])
        self.Sigmoid1 = nn.ReLU()

        self.DenseLayer2 = nn.Linear(self.hiddens[0],self.hiddens[1])
        self.Sigmoid2 = nn.ReLU()
        self.outlayer = nn.Linear(self.hiddens[1],out_size)
        self.softmax = nn.Softmax(dim=1)

    def convblock(self, in_channels, out_channels, kernel_size, pool_width=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.AvgPool2d(pool_width),
            nn.ReLU(),
        )
    
    def reduce_pool(self):
        # only works for square pooling
        # also only works with "same" padding
        dimreduce = []
        dimreduce.append([self.n_filters[0],self.inputshape1[2]//self.pool_width[0],self.inputshape1[3]//self.pool_width[0]])

        for i in range(len(self.pool_width)-1):
            dimreduce.append([self.n_filters[i+1],dimreduce[i][1]//self.pool_width[i+1],dimreduce[i][2]//self.pool_width[i+1]])

        return dimreduce[-1]

    def forward(self, x,x2):
        # Contracting path

        convblock1 = self.conv1(x)
        convblock2 = self.conv2(convblock1)
        bottleneck = self.conv3(convblock2)


        #flatten convolutions
        bottleneckflat = self.flatten(bottleneck)
        bottleneckandone = torch.cat([bottleneckflat,x2],dim=1)
        # print(bottleneckandone.size())

        # pass to dense layers
        Dense1 = self.DenseLayer1(bottleneckandone)
        Dense1 = self.Sigmoid1(Dense1)


        Dense2 = self.DenseLayer2(Dense1)
        Dense2 = self.Sigmoid2(Dense2)

        outfull = self.outlayer(Dense2)
        outsoftmax = self.softmax(outfull)

        return outsoftmax
    
class CNNregression(nn.Module):
    def __init__(self, in_data1, in_data2, out_data):
        super(CNNregression, self).__init__()

        out_size = out_data.size(-1)
        self.inputshape1 = in_data1.size()
        self.inputshape2 = in_data2.size()
        self.n_filters = [16,16,16]
        self.kernel_size = [3,3,3]
        self.pool_width = [2,2,2]
        self.hiddens = [100,50]

        self.conv1 = self.convblock(self.inputshape1[1],self.n_filters[0],self.kernel_size[0],self.pool_width[0])

        self.conv2 = self.convblock(self.n_filters[0],self.n_filters[1],self.kernel_size[1],self.pool_width[1])

        self.conv3 = self.convblock(self.n_filters[1],self.n_filters[2],self.kernel_size[2],self.pool_width[2])

        self.flatten = nn.Flatten()
        flatdims = self.reduce_pool()
        flatsize = np.prod(flatdims)

        self.DenseLayer1 = nn.Linear(flatsize+self.inputshape2[1],self.hiddens[0])
        self.Sigmoid1 = nn.ReLU()

        self.DenseLayer2 = nn.Linear(self.hiddens[0],self.hiddens[1])
        self.Sigmoid2 = nn.ReLU()
        self.outlayer = nn.Linear(self.hiddens[1],out_size)
        # self.softmax = nn.Softmax(dim=1)

    def convblock(self, in_channels, out_channels, kernel_size, pool_width=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.AvgPool2d(pool_width),
            nn.ReLU(),
        )
    
    def reduce_pool(self):
        # only works for square pooling
        # also only works with "same" padding
        dimreduce = []
        dimreduce.append([self.n_filters[0],self.inputshape1[2]//self.pool_width[0],self.inputshape1[3]//self.pool_width[0]])

        for i in range(len(self.pool_width)-1):
            dimreduce.append([self.n_filters[i+1],dimreduce[i][1]//self.pool_width[i+1],dimreduce[i][2]//self.pool_width[i+1]])

        return dimreduce[-1]

    def forward(self, x,x2):
        # Contracting path

        convblock1 = self.conv1(x)
        convblock2 = self.conv2(convblock1)
        bottleneck = self.conv3(convblock2)


        #flatten convolutions
        bottleneckflat = self.flatten(bottleneck)
        bottleneckandone = torch.cat([bottleneckflat,x2],dim=1)
        # print(bottleneckandone.size())

        # pass to dense layers
        Dense1 = self.DenseLayer1(bottleneckandone)
        Dense1 = self.Sigmoid1(Dense1)


        Dense2 = self.DenseLayer2(Dense1)
        Dense2 = self.Sigmoid2(Dense2)

        outfull = self.outlayer(Dense2)
        # outsoftmax = self.softmax(outfull)

        return outfull
    
class CNNclassifier(nn.Module):
    def __init__(self,in_data1, in_data2, out_size):
        super(CNNclassifier, self).__init__()
        
        # out_size = out_data.size(-1)
        self.inputshape1 = in_data1.size()
        self.inputshape2 = in_data2.size()
        num_conv_blocks = 4
        num_filters = 32
        hiddens = [1000]
        
        self.conv_layers = nn.ModuleList()

        # filterlist.append(num_filters)
        self.conv_layers.append(nn.Conv2d(in_channels=self.inputshape1[1], out_channels=num_filters, kernel_size=3, padding=1))
        # self.conv_layers.append(nn.BatchNorm2d(num_filters))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(num_conv_blocks-1):

            self.conv_layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1))
            # self.conv_layers.append(nn.BatchNorm2d(num_filters))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.flatten = nn.Flatten()
        
        self.linear_layers = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features=self.get_flatten_size()+self.inputshape2[1], out_features=hiddens[0]))
        # self.linear_layers.append(nn.BatchNorm1d(num_features=hiddens[0]))
        self.linear_layers.append(nn.ReLU())
        
        for i in range(len(hiddens)-1):   
            self.linear_layers.append(nn.Linear(in_features=hiddens[i], out_features=hiddens[i+1]))
            # self.linear_layers.append(nn.BatchNorm1d(num_features=hiddens[i+1]))  
            self.linear_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(in_features=hiddens[-1], out_features=out_size)
        self.softmax = nn.Softmax(dim=1)
        
    def get_flatten_size(self):
        sample = torch.randn(1, self.inputshape1[1], self.inputshape1[2], self.inputshape1[3])
        with torch.no_grad():
            x = sample
            for layer in self.conv_layers:
                x = layer(x)
        return x.view(x.size(0), -1).size(1)
        
    def forward(self, x1,x2):
        x = self.conv_layers[0](x1)
        for clayer in self.conv_layers[1:]:
            x = clayer(x)
        x = self.flatten(x)
        x = torch.cat((x, x2), dim=1)
        for dlayer in self.linear_layers:
            x = dlayer(x)
        x = self.output_layer(x)
        out = self.softmax(x)
        return out
