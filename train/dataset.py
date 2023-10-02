from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torch

class DatasetSimulationImages(Dataset): 
    """
    Class of a CNN-based encoder.
    '''

    Attributes
    ----------
    image_path_lists : list
        a list of image paths
    transform : torchvision.transforms
        image transformation composition

    Methods
    ------
    __len__():
        Return the length of the dataset.

    __len__(idx):
        Return the image of the dataset given an index.

    """

    def __init__(self, image_paths_list, transform=None):
        """
        Contructor.

        Parameters
        ----------
        image_path_lists : list
            a list of image paths
        transform : torchvision.transforms
            image transformation composition

        """

        self.image_paths = image_paths_list
        self.transform = transform

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns
        -------
        int

        """
        
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image of the dataset.
        
        Parameters
        ----------
        idx : int
            index of the dataset image
        
        Returns
        -------
        torch.tensor
        
        """
        
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = torchvision.transforms.ToTensor(image)

        if self.transform is not None:
            image = self.transform(image)

        return image
