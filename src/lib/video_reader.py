from numpy import ndarray
from pathlib import Path
import cv2


class VideoReader:
    """A video file reading service."""

    def __init__(self, video_path: Path) -> None:
        """Construct.

        :param video_path: The path to the video file.
        """

        self.__video_capture = cv2.VideoCapture(video_path.as_posix())

        if not self.__video_capture.isOpened():
            raise Exception('Error opening video stream or file')

    def __release_video(self) -> None:
        """Releases the video."""

        self.__video_capture.release()

    def read_video_frames(self, limit: int = None) -> tuple[int, ndarray]:
        """Reads the video frames.

        :param limit: If specified, reads the video frames up until the limit.
        :yields: The video frame number and the video frame.
        """

        frame_number = 0
        while (self.__video_capture.isOpened()):
            ret, frame = self.__video_capture.read()
            frame_number += 1
            if limit and frame_number > limit:
                break
            if ret:
                yield frame_number, frame
            else:
                break

        self.__release_video()
