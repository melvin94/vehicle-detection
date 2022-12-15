# Vehicle detection
A proof of concept computer vision project that investigates vehicle detection.

# Project setup
The following outlines the set up requirements:
1. Pre-requisite: Install Python.
2. Pre-requisite: Install Conda for additional package environment management.
3. Create the development environment and install all the required packages.

## Steps 1-2: Pre-requisites Python & Conda
For the purposes of this project, the pre-requisite steps 1-2 will not be outlined in this document. The set up of these requirements are trivial, but may also require machine specific steps in order to set up correctly. Refer to the [Python Docs](https://www.python.org/downloads/) and the [Conda Docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for instructions on how to set up the pre-requisites for this project.

## Step 3: Create the development environment and install all the required packages
To create the development environment with conda, follow these steps:
1. In a terminal of your choosing, create a conda environment with python 3.9.
    ```shell
    conda create -n tf-py39 python=3.9 -y
    ```
2. In your terminal, activate the conda environment created in the previous Step 3.1.
    ```shell
    conda activate tf-py39
    ```
3. From within the same terminal, navigate to this project folder.
    ```shell
    cd /your/path/to/vehicle-tracking/
    ```
4. Install the project package requirements.
    ```shell
    pip install -r requirements.txt
    ```

# Running the code
This project uses open-source Tensorflow pretrained models for object detection. A list of all available object detection models can be found on [TensorFlowHub](https://tfhub.dev/s?module-type=image-object-detection).

Note: When selecting a pretrained model, be sure to check the class information that the model was trained on. This project uses `mobilenetv2`, which includes detection on some vehicle class categories. The follow class categories are used in this project:
- `Car`
- `Motorcycle`
- `Bus`
- `Truck`

The selected classes can be defined in the following constants file:
```
src/lib/class_constants.py
```

## Video inference
To run vehicle detection on a video, run the `video_inference.py` script with the necessary arguments. An example:
```shell
python src/video_inference.py --model-url https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1 --video-path data/datasets/video/urban-tracker/sherbrooke_video.avi --frame-limit 10
```

## Image inference
To run vehicle detection on a single image, run the `single_image_inference.py` script with the necessary arguments. An example:
```shell
python src/single_image_inference.py --model-url https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1 --image-path data/datasets/image/pascal-voc-2012-val-subset/2008_002875_jpg.rf.32e44b678c9caaf122e8b1dcdb2a11e0.jpg
```

## Image directory inference
To run vehicle detection on a directory of images, run the `image_dir_inference.py` script with the necessary arguments. An example:
```shell
python src/image_dir_inference.py --model-url https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1 --images-path data/datasets/image/pascal-voc-2012-val-subset/
```
