## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## I made the last layer with output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Here is an example
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = (1, 32, 5)
        ## I have also included some more layers like:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        #1st Layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout2d(p=0.1)#It drops 10% of the nodes in CNN randomly to reduce overfitting
        
        
        #2nd Layer
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)#We will normalize our input value in data preprocessing phase, so to get similar output I have used batch normalisation
        self.drop2 = nn.Dropout2d(p=0.2)

        #3rd Layer
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.3)

        #4th Layer
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(p=0.4)


        #5th layer(Linear layer)
        self.fc1 = nn.Linear(25600, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.drop5 = nn.Dropout(p=0.5)
        
        #6th Layer(Linear Layer)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)

        #Final output Layer
        self.final = nn.Linear(1024, 136)

        #Maxpool Layer of krenal (2,2) to reduce overfit
        self.pool1 = nn.MaxPool2d(2, 2)

        
    def forward(self, x):
        
        # x is the input image
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        #our image input first goes into convolution layer giving some output values, then these output value go through a non linear activation function(I have used relu function in this case)
        #then it go into maxpool layer to reduce some over fitting and final it will go to dropout layer where we will drop significant amount of nodes from CNN
        
        #We will do same as above with remaining layers
        x = self.drop2(self.pool1(F.relu(self.conv2_bn(self.conv2(x)))))
        

        x = self.drop3(self.pool1(F.relu(self.conv3_bn(self.conv3(x)))))


        x = self.drop4(self.pool1(F.relu(self.conv4_bn(self.conv4(x)))))


        x = x.view(x.size(0), -1)#To flattern our output array

        x = self.drop5(F.leaky_relu(self.fc1_bn(self.fc1(x))))

        x = self.drop5(F.leaky_relu(self.fc2_bn(self.fc2(x))))

        x =self.final(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
