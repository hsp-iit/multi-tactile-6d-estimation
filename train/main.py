import configparser
import numpy as np
import torch
import glob
import sys
import torchvision

from encoder import Encoder
from decoder import Decoder
from dataset import DatasetSimulationImages
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# Make sure the training is reproducible 
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# Parse the config.ini file
config = configparser.ConfigParser()
config_file = sys.argv[1]
config.read(config_file)
paths = config['Paths']
normalization = config['Normalization']
hyperparameters = config['Hyperparameters']

# Load the paths
data_path = paths['dataset']
prism_images_path = paths['prism_images_path']
model_name = paths['model_name']

# Load the normalization parameters
normalization_bool = normalization['boolean']
mean = eval(normalization['mean'])
std = eval(normalization['std'])
mean_tensor = torch.FloatTensor(mean)
std_tensor = torch.FloatTensor(std)

# Load the hyperparameters.
batch_size = int(hyperparameters['batch_size'])
encoded_space = int(hyperparameters['encoded_space'])
bootstrap_ratio = int(hyperparameters['bootstrap_ratio'])
lr = float(hyperparameters['learning_rate'])

# Loop over the data
image_paths = []
for data_path_loop in glob.glob(data_path + '/*'):
    image_paths.append(data_path_loop)

image_paths.sort()

# Split the data in train, valid and test sets
m = len(image_paths)
train_image_paths, valid_image_paths, test_image_paths = random_split(image_paths,
                                                                    [int(0.7 * m), int(0.2 * m),
                                                                    int(m - int(0.7 * m) - int(0.2 * m))],
                                                                    generator=torch.Generator().manual_seed(0))

# Set the necessary transforms for the datasets
transforms = None

if normalization_bool == 'True':
    transforms = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])
    print("Normalization active")

# Create the train, validation and test sets
train_dataset = DatasetSimulationImages(image_paths_list=train_image_paths,
                                        transform=transforms)
valid_dataset = DatasetSimulationImages(image_paths_list=valid_image_paths,
                                        transform=transforms)
test_dataset = DatasetSimulationImages(image_paths_list=test_image_paths,
                                       transform=transforms)

# Initialize the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, generator=g)

# Initialize the encoded space
encoder = Encoder(image_size_W=240, image_size_H=320, latent_size=encoded_space)
decoder = Decoder(image_size_W=240, image_size_H=320, latent_size=encoded_space)

# Initialize the parameters to optimize
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Parallelize the models and move both the encoder and the decoder to the selected device.
encoder = torch.nn.DataParallel(encoder)
decoder = torch.nn.DataParallel(decoder)
encoder.to(device)
decoder.to(device)

def loss_aae(x, x_hat, bootstrap_ratio):
    if bootstrap_ratio > 1:
        mse = torch.flatten((x_hat - x) ** 2)
        loss_aae = torch.mean(torch.topk(mse, mse.numel() // bootstrap_ratio)[0])
    else:
        loss_aae = torch.nn.functional.mse_loss(x, x_hat)
    return loss_aae

# Training function.
def train_epoch(encoder, decoder, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder.
    encoder.train()
    decoder.train()
    train_loss = []
    for i, image_batch in enumerate(tqdm(dataloader)):
        # Move tensor to the proper device, encode and decode the data
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        # Evaluate loss.
        loss = loss_aae(image_batch, decoded_data, bootstrap_ratio)

        # Backward pass.
        # Set the gradient of the optimized tensors to zero so that they are not touched again.
        optimizer.zero_grad()

        # Compute the loss.
        loss.backward()
        
        # Update the parameters.
        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())
        

    return np.mean(train_loss)


# Testing function.
def test_epoch(encoder_test, decoder_test, device_test, dataloader_test):
    # Set evaluation mode for encoder and decoder.
    # In this way, batch normalization layers will work in eval mode.
    encoder_test.eval()
    decoder_test.eval()
    # Since we are in inference, we do not want to calculate the gradient.
    with torch.no_grad():
        # Define the lists to store the output tensors for each batch.
        decoded_images = []
        starting_images = []

        for image_batch in dataloader_test:
            # Move tensor to the proper device, encode and decode the data.
            image_batch = image_batch.to(device_test)
            encoded_data = encoder_test(image_batch)
            decoded_data = decoder_test(encoded_data)

            # Append the network output and the original image to the lists and move the images back to the cpu.
            decoded_images.append(decoded_data.cpu())
            starting_images.append(image_batch.cpu())

        # Create a single tensor with all the values in the lists.
        decoded_images = torch.cat(decoded_images)
        starting_images = torch.cat(starting_images)

        # Evaluate global loss.
        val_loss = loss_aae(decoded_images, starting_images)

    return val_loss.data


num_epochs = 21
history = {'train_loss': [], 'validate_loss': []}

# Loop over the epochs
for epoch in range(num_epochs):
    # Train and validate the model
    train_loss = train_epoch(encoder, decoder, device, train_loader, optim)
    validate_loss = test_epoch(encoder, decoder, device, valid_loader)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
                                                                          validate_loss))
    history['train_loss'].append(train_loss)
    history['validate_loss'].append(validate_loss)

    # Save the model
    if epoch % 4 == 0:
        torch.save({"model_state_dict_encoder": encoder.state_dict(), "model_state_dict_decoder": decoder.state_dict()},
                   model_name + str(epoch) + '.pth')
