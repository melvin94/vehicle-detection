from argparse import ArgumentParser
from glob import glob
from lib.tf_prediction_utils import bounding_box, image_to_tensor
from os import makedirs
from os.path import join
from pathlib import Path
from tqdm import tqdm as progress_bar

import cv2
import tensorflow_hub as hub

# construct an argument parser
ap = ArgumentParser()
ap.add_argument(
    '-m',
    '--model-url',
    required=True,
    help='Specify the url to the tensorflow model.'
)
ap.add_argument(
    '-i',
    '--images-path',
    required=True,
    help='Specify the directory to the folder containing images for inference.'
)
ap.add_argument(
    '-o',
    '--output-path',
    required=False,
    help='Specify the output path to save the inference results.'
)
ap.add_argument(
    '-s',
    '--superclass',
    required=False,
    default='vehicle',
    help='Specify the superclass to filter the prediction results with.'
)

if __name__ == '__main__':
    args = ap.parse_args()
    model_url = args.model_url
    images_path = Path(args.images_path)
    superclass = args.superclass

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        folder_name = images_path.stem
        output_path = join('data', 'outputs', 'image-dir-inf', folder_name)
    makedirs(output_path, exist_ok=True)

    detector = hub.load(model_url).signatures['default']

    image_search_path = glob(join(images_path, '*'))
    for image_path in progress_bar(image_search_path):
        image = cv2.imread(image_path)

        prediction = detector(image_to_tensor(image))

        _, image = bounding_box(image, prediction, superclass=superclass)

        filename = Path(image_path).stem
        output_image_path = join(output_path, f'{filename}.png')
        cv2.imwrite(output_image_path, image)
