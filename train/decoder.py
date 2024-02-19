import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder(nn.Module):
    """
    Class of a CNN-based decoder.
    '''

    Attributes
    ----------
    deconvs : nn.ModuleList
        a list of deconvolutional layers
    fc : nn.Linear
        fully connected layer
    flatten : nn.Flatten
        a flatten layer
    image_size_h : int
        height of the output image
    image_size_w : int
        width of the output image
    last_filter : int
        number of kernels in the last layer
    stride factor : int
        factor of the total strides
    ups : nn.ModuleList
        a list of upsample layers

    Methods
    ------
    forward(x):
        Forward pass of the network. It reconstructs the output image given the latent representation x.

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
            width of the output image
        image_size_h : int
            height of the output image
        latent_size : int
            size of the latent space
        filters : tuple
            number of filters
        conv : tuple
            kernel sizes
        stride : tuple
            stride sizes

        """

        super().__init__()

        # Set the parameters of the network
        in_channels = list(reversed(filters))
        out_channels = in_channels[1:] + [3]
        conv = list(reversed(conv))
        stride = list(reversed(stride))

        # Intiialize the deconvolutional layers
        self.deconvs = nn.ModuleList([
            nn.Conv2d(in_channels=ic,
                      out_channels=oc,
                      kernel_size=c,
                      padding=c // 2) for ic, oc, c, s in zip(in_channels, out_channels, conv, stride)
        ])

        # Initialize the upsample layers
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=s, mode='nearest') for s in stride])

        # Useful just because the weights need this field
        self.bns = nn.ModuleList([nn.BatchNorm2d(oc) for oc in out_channels[:-1]])
        # Initialize the flatten layer
        self.flatten = nn.Flatten()

        # Compute the stride factor
        self.stride_factor = 1
        for s in stride:
            self.stride_factor *= s

        # Assign the dimensions of the output
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.last_filter = in_channels[0]

        # Initialize the first layer
        output_linear_size = int(self.last_filter * (image_size_w/ self.stride_factor) *
                                (image_size_h/ self.stride_factor))
        self.fc = nn.Linear(latent_size, output_linear_size)


    def forward(self, x):
        """
        Forward pass of the network. It reconstructs the output image given the latent representation x.

        Parameters
        ----------
        x : torch.tensor
            tensor of the input latent vectors

        Returns
        -------
        torch.tensor

        """

        # Forward the image through the network
        x = self.fc(x)
        x = F.relu(x)
        x = x.view((-1, self.last_filter, self.image_size_h // self.stride_factor, self.image_size_w // self.stride_factor))

        for conv, up in zip(self.deconvs[:-1], self.ups[:-1]):
            x = up(x)
            x = conv(x)
            x = F.relu(x)

        x = self.ups[-1](x)
        x = self.deconvs[-1](x)

        return torch.nn.LeakyReLU()(x)
