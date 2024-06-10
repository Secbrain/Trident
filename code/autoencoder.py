import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Autoencoder1(nn.Module):
    def __init__(self, y_in, y1, y2, y3, y4):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(y_in, y1),
            nn.ReLU(inplace=True),
            nn.Linear(y1, y2),
            nn.ReLU(inplace=True),
            nn.Linear(y2, y3),
            nn.ReLU(inplace=True),
            nn.Linear(y3, y4),
            #nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(y4, y3),
            nn.ReLU(inplace=True),
            nn.Linear(y3, y2),
            nn.ReLU(inplace=True),
            nn.Linear(y2, y1),
            nn.ReLU(inplace=True),
            nn.Linear(y1, y_in),
            #nn.Sigmoid()
        )
        #self.soft = nn.Softmax(dim = 0)

    def forward(self, x):
        x = self.encoder(x)
        #print(x.shape, '-'*3, type(x))
        #print(tags.shape, '-'*3, type(tags))
        x = self.decoder(x)
        #x = self.soft(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, y_in, y1, y2, y3, y4):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(nn.Linear(y_in, y1), nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(nn.Linear(y1, y2), nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(nn.Linear(y2, y3), nn.ReLU(inplace=True))
        self.encoder4 = nn.Linear(y3, y4)

        self.decoder1 = nn.Sequential(nn.Linear(y4, y3), nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Linear(y3, y2), nn.ReLU(inplace=True))
        self.decoder3 = nn.Sequential(nn.Linear(y2, y1), nn.ReLU(inplace=True))
        self.decoder4 = nn.Sequential(nn.Linear(y1, y_in))

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d1 = self.decoder1(e4)
        d2 = self.decoder2(d1 + e3)
        d3 = self.decoder3(d2 + e2)
        x = self.decoder4(d3 + e1)
        return x

