# Toxic-Comment-Classification

## Requirements for training with GPU (in order of installation)
- Python 3.10.x
- Visual Studio 2019 with "Desktop development with C++" workload
- CUDA 11.2.2 > https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN 8.1.1 > https://developer.nvidia.com/rdp/cudnn-archive
- Add CUDA and cuDNN to PATH
- tensorflow 2.10
- keras 2.10

# Steps
- Clone the repository
- Create a virtual environment
- Install setuptools and wheel
- Install requirements.txt
- Download glove
- Run create_custom_glove.py
- Run main.py