import torch.nn as nn
import torch.nn.functional as F
import torch


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

    def requires_grad(self, state):
        for param in self.parameters():
            param.requires_grad = state


class DepthwiseDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(DepthwiseDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.comb1 = nn.Conv2d(num_classes, ndf, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, groups=ndf)
        self.comb2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=2, padding=1, groups=ndf*2)
        self.comb3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1, groups=ndf*4)
        self.comb4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.comb1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.comb2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.comb3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.comb4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

    def requires_grad(self, state):
        for param in self.parameters():
            param.requires_grad = state
