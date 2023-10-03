import configparser
import numpy as np
import sys
import torch
import torchvision

from PIL import Image
from train.encoder import Encoder
from train.decoder import Decoder

class TactileBasedSelector():
    """
    A class to hadle the tactile-based selection of the points.
    

    Attributes
    ----------
    angles_database : str
        name of the angles database file
    images_point_cloud : str
        filenames of the images
    transforms : torchvision.transforms.Compose
        composition of transformations for the images
    angles_database : str
        filename of the database
    encoder : Encoder
        CNN-based encoder
    decoder : Decoder
        CNN-based Decoder
    device : torch.device
        device where the model run
    poses_array : np.array
        array of points
    number_of_sensors : int
        number of sensors used
    angles_comparison_vectors : list
        list of angles between the background and the inference images
    point_clouds_array : list
        list of arrays containing the partial point clouds
    indexes_list : list
        list of arrays contaning the indexes of the partial point clouds
    latent_vector_background : torch.tensor
        tensor with the latent vector of the background image

    Methods
    ------
    calculate_indexes():
        Calculate the remaining indexes.
    calcuate_database():
        Calculate the off-line database of the images w.r.t. the background.
    
    """
    
    def __init__(self, config_file_path : str)-> None:
        """
        Constructor

        Args:
            config_file_path (str): _description_
        """
        
        # Set the path to the folder containing the images
        config = configparser.ConfigParser()
        config.read(config_file_path)

        # Parse the file
        autoencoder = config['Autoencoder']
        eliminate_points = config['EliminatePoints']
        self.angles_database = eliminate_points['angles_database']
        self.images_point_cloud = eliminate_points['images_point_cloud']
        # Assign the useful values
        normalization_bool = autoencoder['normalization_bool']
        mean = eval(autoencoder['mean'])
        std = eval(autoencoder['std'])

        encoded_space = int(autoencoder['encoded_space'])

        # Set the necessary transforms for the datasets
        self.transforms = None
        if normalization_bool == 'True':
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])

        # Initialize the encoded space

        self.encoder = Encoder(image_size_W=240, image_size_H=320, latent_size=encoded_space)
        self.decoder = Decoder(image_size_W=240, image_size_H=320, latent_size=encoded_space)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        # Parallelize the models and move both the encoder and the decoder to the selected device.
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Load the weights to the model
        model = torch.load(autoencoder['model'])
        self.encoder.load_state_dict(model['model_state_dict_encoder'])
        self.decoder.load_state_dict(model['model_state_dict_decoder'])

        # Load the point cloud of the object
        self.poses_array = np.loadtxt(eliminate_points['point_cloud_file'])
        self.number_of_sensors = int(eliminate_points['number_of_sensors'])
        background = torchvision.transforms.ToTensor()(Image.open(autoencoder['background']))
        background = self.transforms(background)
        background = background.to(self.device)
        self.angles_comparison_vectors = []
        self.point_clouds_array = []
        self.indexes_list = []

        with torch.no_grad():
            
            # Calculate the latent vector of the background
            self.latent_vector_background = self.encoder(background.unsqueeze(0))
            
            # Calculte the angles of the inference images
            for i in range(self.number_of_sensors):
                comparison_image = torchvision.transforms.ToTensor()(Image.open(eliminate_points['image_sensor_'+str(i+1)]))
                comparison_image = self.transforms(comparison_image)
                comparison_vector = self.encoder(comparison_image.unsqueeze(0))
                angle_comparison_vector = torch.acos(torch.matmul(self.latent_vector_background, torch.t(comparison_vector))
                               / (torch.linalg.norm(self.latent_vector_background)*torch.linalg.norm(comparison_vector))).item()
                print(angle_comparison_vector)
                self.angles_comparison_vectors.append(angle_comparison_vector)
                self.point_clouds_array.append(np.empty((0, 6)))
                self.indexes_list.append(np.empty((0,1)))


    def calculate_indexes(self) -> None:
        """
        Calculate the remaining indexes
        """

        angles = np.loadtxt(self.angles_database)

        # Save the indexes
        for i in range(self.poses_array.shape[0]):
            for j in range(self.number_of_sensors):

                if abs(self.angles_comparison_vectors[j] - angles[i]) < 0.04:
                    self.point_clouds_array[j] = np.append(self.point_clouds_array[j], np.array([self.poses_array[i]]), 0)
                    self.indexes_list[j] = np.append(self.indexes_list[j], np.array([[i]]), 0)

        # Save the partial point cloud for visualization purpose
        for j in range(self.number_of_sensors):

            f = open("heatmap" + str(j+1) + ".off", 'w')
            f.write("COFF\n")
            f.write(str(self.point_clouds_array[j].shape[0]) + " " + str(0) + " " + str(0) + "\n")
            f.close()


            for i in range(self.point_clouds_array[j].shape[0]):
                f = open("heatmap"+str(j+1)+".off", "a")
                f.write(str(self.point_clouds_array[j][i][0]) + " " + str(self.point_clouds_array[j][i][1]) + " " + str(self.point_clouds_array[j][i][2]) + " " + str(1.0) + " " + str(0.0) + " " + str(0.0) + " " + str(1.) + "\n")


    def calculate_database(self) -> None:
        """
        Calculate the off-line database of the images w.r.t. the background
        """

        angles = []
        for i in range(self.poses_array.shape[0]):
            image = torchvision.transforms.ToTensor()(Image.open(self.images_point_cloud + "Image_heatmap_" + str(i) + ".png"))
            if self.transforms is not None:
                image = self.transforms(image)

            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                latent_vector = self.encoder(image.unsqueeze(0).to(self.device))
            angle = torch.acos(torch.matmul(self.latent_vector_background, torch.t(latent_vector))
                               / (torch.linalg.norm(self.latent_vector_background)*torch.linalg.norm(latent_vector))).item()

            angles.append(angle)
        angles = np.array(angles)
        np.savetxt(self.angles_database, angles)



if __name__ == '__main__':
    
    config_file = sys.argv[1]
    init = TactileBasedSelector(config_file)

    init.calculate_indexes()
