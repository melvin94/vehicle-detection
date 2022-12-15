from argparse import ArgumentParser
from lib.video_reader import VideoReader
from pathlib import Path

import cv2

# construct an argument parser
ap = ArgumentParser()
ap.add_argument(
    '-v',
    '--video-path',
    required=True,
    help='Specify the path to the video to load.'
)
ap.add_argument(
    '-f',
    '--frame-limit',
    required=False,
    type=int,
    help='Specify the video frame limit to read up until.'
)

if __name__ == '__main__':
    args = ap.parse_args()
    video_path = Path(args.video_path)
    frame_limit = args.frame_limit

    video_reader = VideoReader(video_path)

    for _, frame in video_reader.read_video_frames(limit=frame_limit):
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closes all the frames
    cv2.destroyAllWindows()
