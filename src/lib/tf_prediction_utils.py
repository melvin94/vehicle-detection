from lib.class_constants import CLASS_METADATA, SUPERCLASSES
from lib.image_utils import draw_box_with_text

import numpy as np
import tensorflow as tf


def image_to_tensor(image: np.ndarray) -> tf.float32:
    """Converts an image to a Tensor."""

    image_np_expanded = np.expand_dims(image, axis=0)

    return tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)


def __get_prediction_data(image: np.ndarray,
                          box: tf.Variable,
                          class_entity: tf.Variable,
                          score: tf.Variable) -> tuple[tuple, str, int]:
    """Gets the prediction data from an inference output for specific outputs.

    :param image: The image that inference was done on.
    :param box: The bounding box outputs from the inference results.
    :param class_entity: The class names outputs from the inference results.
    :param score: The prediction confidence outputs from the inference results.
    :returns: A standardized format of `box`, `class_entity`, and `score`.
    """

    imageH, imageW = image.shape[:2]

    y1, x1, y2, x2 = box.numpy()
    x1 = int(x1*imageW)
    x2 = int(x2*imageW)
    y1 = int(y1*imageH)
    y2 = int(y2*imageH)
    box_rect = (x1, y1, x2, y2)

    class_name = class_entity.numpy().decode('ascii')

    score = score.numpy()

    return box_rect, class_name, score


def bounding_box(image: np.ndarray,
                 prediction: dict,
                 min_score: int = 0.1,
                 superclass: str = None) -> tuple[bool, np.ndarray]:
    """Places bounding boxes on the inferenced image based on the predictions.

    :param prediction: The inference outputs.
    :param min_score:  The minimum prediction confidence score for a positive
                       detection.
    :param superclass: Filter predicted classes by a superclass defined in
                       `class_constants`.
    :returns:          A tuple including a boolean result that determines if a
                       positive detection was found, and the image with
                       bounding boxes for any predictions found.
    """

    boxes = prediction['detection_boxes']
    class_entities = prediction['detection_class_entities']
    scores = prediction['detection_scores']

    detection = False
    for i, box in enumerate(boxes):
        box_rect, class_name, score = __get_prediction_data(
            image,
            box,
            class_entities[i],
            scores[i]
        )

        if score < min_score:
            continue
        if superclass and class_name not in SUPERCLASSES[superclass]:
            continue

        detection = True

        text_box_str = f'{class_name}: {int(score*100)}%'
        color = CLASS_METADATA[class_name]['color']
        image = draw_box_with_text(image, box_rect, text_box_str, color)

    return detection, image
