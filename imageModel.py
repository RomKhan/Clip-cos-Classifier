import torch
from torch import nn

class ImageModel(torch.nn.Module):

    def __init__(self):
        super(ImageModel, self).__init__()

        self.linear1 = torch.nn.Linear(768, 200)
        self.norm = torch.nn.BatchNorm1d(200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 2)

        self.linear1.weight.data.uniform_(-1, 1)
        self.linear2.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x