from argparse import ArgumentParser
from lib.tf_prediction_utils import bounding_box, image_to_tensor
from lib.video_reader import VideoReader
from os import makedirs
from os.path import join
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
    '-v',
    '--video-path',
    required=True,
    help='Specify the path to the video to convert to frames.'
)
ap.add_argument(
    '-o',
    '--output-path',
    required=False,
    help='Specify the output path to save the inference results.'
)
ap.add_argument(
    '-f',
    '--frame-limit',
    required=False,
    type=int,
    help='Specify the video frame limit to read up until.'
)
ap.add_argument(
    '-s',
    '--superclass',
    required=False,
    default='vehicle',
    help='Specify the superclass to filter the prediction results with.'
)
ap.add_argument('--viewer-mode', action='store_true')

if __name__ == '__main__':
    args = ap.parse_args()
    model_url = args.model_url
    video_path = Path(args.video_path)
    superclass = args.superclass
    frame_limit = args.frame_limit
    viewer_mode = args.viewer_mode

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        folder_name = video_path.stem
        output_path = join('data', 'outputs', 'video-inf-frames', folder_name)

    makedirs(output_path, exist_ok=True)

    detector = hub.load(model_url).signatures['default']

    video_reader = VideoReader(video_path)
    for frame_num, frame in video_reader.read_video_frames(limit=frame_limit):
        print(f'Processing frame-{frame_num} ...')
        prediction = detector(image_to_tensor(frame))

        detection, img = bounding_box(frame, prediction, superclass=superclass)

        if detection:
            output_frame_path = join(output_path, f'frame-{frame_num}.png')
            cv2.imwrite(output_frame_path, img)

        if viewer_mode:
            # Display the resulting frame
            cv2.imshow('Frame', img)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if viewer_mode:
        # Closes all the frames
        cv2.destroyAllWindows()
