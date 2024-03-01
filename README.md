<h1 align="center">
    Object pose estimation <br> using multiple vision-based tactile sensors
</h1>

<p align="center"><img src="https://user-images.githubusercontent.com/49904924/235861528-2cd16f61-2b4f-4764-a805-b22a1541f478.png" | width =600 alt=""/></p>


<h4 align="center">
  Collision-aware In-hand 6D Object Pose Estimation using Multiple Vision-based Tactile Sensors
</h4>

<div align="center">
  2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 719-725
</div>

<div align="center">
  <a href="https://ieeexplore.ieee.org/document/10160359"><b>Paper</b></a> |
  <a href="https://www.youtube.com/watch?v=joR0Yp1zQ_U"><b>Video</b></a>
</div>

## Table of Contents

- [Update](#new-updates)
- [Installation](#gear-installation)
- [Citing this paper](#-citing-this-paper)


## :gear: Installation

### Strict Requirements
- [pytorch](https://pytorch.org/)
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
- [jax](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)
```
pip install jaxlie jax==0.3.4
```
- jaxlib
```
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.2+cuda11.cudnn82-cp38-none-manylinux2010_x86_64.whl
```
- [NVIDIA PhysX SDK 4.1](https://github.com/NVIDIAGameWorks/PhysX)
### Dependencies
- numpy==1.22.3
- scipy==1.8.0
- pyquaternion
- tqdm

## How to install
### No docker
#### 1. Install NVIDIA PhysX SDK 4.1
Clone the [repo](https://github.com/NVIDIAGameWorks/PhysX.git) and follow the [instructions](https://github.com/NVIDIAGameWorks/PhysX?tab=readme-ov-file#quick-start-instructions). For linux installation, you need to install clang:

```
sudo apt-get install clang
```

In order to let CMAKE use it as compiler, you should export the CMAKE_CXX_COMPILER environment variable in this way:

```
export CMAKE_CXX_COMPILER=/usr/bin/clang
```

```
sudo apt-get install libxxf86vm-dev
```

#### 2. Clone repository
```
git clone https://github.com/hsp-iit/multi-tactile-6d-estimation.git 
cd multi-tactile-6d-estimation
```

Then, you need to compile the collision detection code.

``` 
cd physX 
mkdir build
```

To compile the code, you need to set in the [CMakeLists.txt](physX/CMakeLists.txt) file the absolute path to PhysX/physx you compiled in the first step.

```
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug "-GUnix Makefiles" -DCMAKE_CXX_FLAGS=-Wno-restrict -Wno-class-memaccess 
sed -e s/-Werror//g -i /home/user/multi-tactile-6d-estimation/physX/build/externals/physx/sdk_source_bin/CMakeFiles/PhysXExtensions.dir/flags.make
make
```

#### 3. Get the model
```
cd multi-tactile-6d-estimation && apt install git-lfs & git lfs install && git clone https://huggingface.co/gabrielecaddeo/tactile-autoencoder
```

### Docker
You need to build the image provided in [docker](docker) folder by running
```
cd docker
bash run.sh
```

## How to run
To reproduce the experiments, you need to run
```
cd multi-tactile-6d-estimation/optimization
bash run.sh /path/to/multi-tactile-6d-estimation /path/to/multi-tactile-6d-estimation/tactile-autoencoder/weights/model_real_back_norm.pth
```
You will find the table_all_objects.tex in results_directory_pose_estimation directory
## Fix
The rotation metrics' results are expected to be slightly better than those presented in the paper, attributed to a minor error in the [final_results.py](optimization/final_results.py) file

## 📰 Citing this paper

```bibtex
@INPROCEEDINGS{10160359,
  author={Caddeo, Gabriele M. and Piga, Nicola A. and Bottarel, Fabrizio and Natale, Lorenzo},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Collision-aware In-hand 6D Object Pose Estimation using Multiple Vision-based Tactile Sensors}, 
  year={2023},
  volume={},
  number={},
  pages={719-725},
  doi={10.1109/ICRA48891.2023.10160359}}
```

## 🧔 Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/gabrielecaddeo.png" width="40">](https://github.com/gabrielecaddeo) | [@gabrielecaddeo](https://github.com/gabrielecaddeo) |
