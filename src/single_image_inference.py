from argparse import ArgumentParser
from lib.tf_prediction_utils import bounding_box, image_to_tensor
from pathlib import Path

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
    '--image-path',
    required=True,
    help='Specify the image path for inference.'
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
    image_path = Path(args.image_path)
    superclass = args.superclass

    detector = hub.load(model_url).signatures['default']

    image = cv2.imread(image_path.as_posix())
    prediction = detector(image_to_tensor(image))

    detection, image = bounding_box(image, prediction, superclass=superclass)

    window_name = f'Detection found: {detection}'

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
