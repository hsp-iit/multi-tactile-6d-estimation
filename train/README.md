## How to train the network

First, you should put in the [config](config.ini) file the path to the dataset folder. Then, you should run:

`CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py /path/to/the/config/file`
### Dataset
The dataset files are going to be uploaded soon. For the time being, you can use the trained [model](https://huggingface.co/gabrielecaddeo/tactile-autoencoder).
