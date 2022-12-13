# Vehicle detection and tracking
A computer vision project the investigates vehicle detection and tracking.

# Project setup
The following outlines the set up requirements:
1. Pre-requisite: Install Python.
2. Pre-requisite: Install Conda for additional package environment management.
3. Pre-requisite: Install NVIDIA CUDA (the models in this project are trained with GPU capability).
4. Create the development environment and install all the required packages.

## Pre-requisites: Python | Conda | CUDA
For the purposes of this project, the pre-requisite steps 1-3 will not be outlined in this document. The set up of these requirements are trivial, but may also require machine specific steps in order to set up correctly. Refer to the [Python Docs](https://www.python.org/downloads/), the [Conda Docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and the [CUDA Docs](https://docs.nvidia.com/cuda/) for instructions on how to set up the pre-requisites for this project.

## Detectron2
This project utilizes Facebook AI Research's [Detectron2](https://github.com/facebookresearch/detectron2) framework for object detection. Refer to the detectron2 official GitHub repository for detailed instructions and guidelines on how to install detectron2 depending on the desired configuration and the corresponding configuration of your machine.

## Create the development environment and install all the required packages
My machine runs on Windows 11, and the following steps were followed to correctly set up detectron2 on my machine with CUDA enabled:
1. In a terminal of your choosing, create a conda environment with python 3.8.
    ```shell
    conda create -n py38 python=3.8 -y
    ```
2. Once the environment has been created, activate the environment and then install the PyTorch packages with CUDA enabled. The package versions used are indicated as follows:
    ```shell
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
3. Install detectron2. The following command is typically used to install the latest version of detectron2:
    ```shell
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```
    However, detectron2 failed to build on my machine when running this command. The following steps were required in order to correctly install detectron2 on my machine:
    1. Clone the detectron2 repo locally, but not in this same project folder. From within the same terminal that you have created your conda environment, navigate to some other directory on your machine, and clone the detectron2 repo:
        ```shell
        git clone https://github.com/facebookresearch/detectron2.git
        ```
    2. Open the following detectron2 file in an editor of your choosing:
        ```
        detectron2\detectron2\layers\csrc\nms_rotated\nms_rotated_cuda.cu
        ```
    3. Replace the following lines in this file as follows:
        ```C
        ...
        #ifdef WITH_CUDA
        #include "../box_iou_rotated/box_iou_rotated_utils.h"
        #endif
        // TODO avoid this when pytorch supports "same directory" hipification
        #ifdef WITH_HIP
        #include "box_iou_rotated/box_iou_rotated_utils.h"
        #endif
        ...
        ```
        replace with
        ```C
        ...
        #include "../box_iou_rotated/box_iou_rotated_utils.h"
        ...
        ```
        The import statements in this file should look as follows:
        ```C
        ...
        #include <ATen/ATen.h>
        #include <ATen/cuda/CUDAContext.h>
        #include <c10/cuda/CUDAGuard.h>
        #include <ATen/cuda/CUDAApplyUtils.cuh>
        #include "../box_iou_rotated/box_iou_rotated_utils.h"
        ...
        ```
        Save the file.
    4. In your terminal, manually install the detectron2 repo as follows:
        ```shell
        pip install -e detectron2
        ```
4. In your terminal, navigate back to this project folder. Install the other project package requirements as follows:
    ```shell
    pip install -r requirements.txt
    ```

# Running the code