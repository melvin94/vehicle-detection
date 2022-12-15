# Data
A subset of publically available open-source datasets are included in this project as sample input data that can be used with the scripts in this project. The data includes an image dataset, and a video dataset.

## Image
The images were acquired from the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset. A subset of ten (10) images were manually sourced from the validation set, including only images containing some form of a vehicle. These images are located in the `image` folder:
```
vehicle-tracking/data/datasets/image/pascal-voc-2012-val-subset/
```

The data was downloaded from [roboflow](https://public.roboflow.com/object-detection/pascal-voc-2012/1).

### Roboflow Citation: 
Pascal VOC 2012 > raw

Provided by [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.html).

License: CC BY 4.0

Pascal VOC 2012 is common benchmark for object detection. It contains common objects that one might find in images on the web.

![Image example](https://i.imgur.com/y2sB9fD.png)

Note: the test set is witheld, as is common with benchmark datasets.

You can think of it sort of like a baby [COCO](https://blog.roboflow.com/coco-dataset/). 

```
This dataset was exported via roboflow.ai on June 18, 2021 at 12:03 PM GMT

It includes 17112 images.
VOC are annotated in Tensorflow Object Detection format.

The following pre-processing was applied to each image:

No image augmentation techniques were applied.
```

## Video
The video was acquired from the [Urban Tracker](https://www.jpjodoin.com/urbantracker/dataset.html) dataset.

### Citation
`vehicle-tracking/data/datasets/video/urban-tracker/sherbrooke_video.avi`:
```
Jodoin, J.-P., Bilodeau, G.-A., Saunier, N., Urban Tracker: Multiple Object Tracking in Urban Mixed Traffic, Accepted for IEEE Winter conference on Applications of Computer Vision (WACV14), Steamboat Springs, Colorado, USA, March 24-26, 2014
```



