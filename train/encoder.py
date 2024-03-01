import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    """
    Class of a CNN-based encoder.
    '''

    Attributes
    ----------
    bns : nn.ModuleList
        a list of batch normalization layers
    convs : nn.ModuleList
        a list of convolutional layers
    fc : nn.Linear
        fully connected layer
    flatten : nn.Flatten
        a flatten layer

    Methods
    ------
    forward(x):
        Forward pass of the network. It provides a latent representation of the input image x.

    """

    def __init__(self, image_size_w=128,
                       image_size_h=128,
                       latent_size=128,
                       filters=(128, 256, 256, 512),
                       conv=(5, 5, 5, 5),
                       stride=(2, 2, 2, 2)):
        """
        Contructor.

        Parameters
        ----------
        image_size_w : int
            width of the input image
        image_size_h : int
            height of the input image
        latent_size : int
            size of the latent space
        filters : tuple
            number of kernels
        conv : tuple
            kernel sizes
        stride : tuple
            stride sizes

        """

        super().__init__()

        # Set the input and output channels for the layers
        out_channels = list(filters)
        in_channels = [3] + out_channels[:-1]

        # Initialize the convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(ic, oc, c, s, c // 2) for ic, oc, c, s in zip(in_channels, out_channels, conv, stride)
        ])

        # Initialize the batch normalization layers
        self.bns = nn.ModuleList([nn.BatchNorm2d(oc) for oc in out_channels[:-1]])

        # Initialize the flatten layer
        self.flatten = nn.Flatten()

        # Compute the stride factor
        stride_factor = 1
        for s in stride:
            stride_factor *= s

        # Get the input size of the fully connecte layer
        input_linear_size = int(filters[-1] * (image_size_w / stride_factor) * (image_size_h / stride_factor))

        # Intiialize the fully connected layer
        self.fc = nn.Linear(input_linear_size, latent_size)


    def forward(self, x):
        """
        Forward pass of the network. It provides a latent representation of the input image x.

        Parameters
        ----------
        x : torch.tensor
            tensor of the images

        Returns
        -------
        torch.tensor

        """

        # Forward the image through the network
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)

        x = self.convs[-1](x)
        x = F.relu(x)
        x = self.flatten(x)

        return self.fc(x)
