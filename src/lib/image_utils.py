from numpy import ndarray

import cv2


def draw_rectangle(image: ndarray,
                   rect_coords: tuple[int, int, int, int],
                   color: tuple[int, int, int],
                   thickness: int) -> ndarray:
    """Draws a rectangle on an image.

    :param image:       The image.
    :param rect_coords: The rectangle coordinates `[x1, y1, x2, y2]` where the
                        (x1, y1) is the top-left corner, and (x2, y2) is the
                        bottom-right corner.
    :param color:       The BGR color of the rectangle.
    :param thickness:   The border thickness of the rectangle.
    :returns:           The image with the drawn rectangle.
    """

    x1, y1, x2, y2 = rect_coords

    return cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


def draw_text(image: ndarray,
              text: str,
              bottom_left_coord: tuple[int, int],
              color: tuple[int, int, int]) -> ndarray:
    """Draws text on an image.

    :param image: The image.
    :param text: The text to draw on the image.
    :param bottom_left_coord: The bottom-left point of the text box (x, y).
    :param color: The BGR color of the rectangle.
    :returns: The image with the drawn text.
    """

    return cv2.putText(
        image,
        text,
        bottom_left_coord,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA
    )


def draw_box_with_text(image: ndarray,
                       box_rect: tuple[int, int, int, int],
                       text: str,
                       color: tuple[int, int, int],
                       box_thickness: int = 4):
    """Draws a bounding box with text on an image.

    :param image:         The image.
    :param rect_coords:   The rectangle coordinates `[x1, y1, x2, y2]` where
                          the (x1, y1) is the top-left corner, and (x2, y2) is
                          the bottom-right corner.
    :param text:          The text to draw on the image.
    :param color:         The BGR color of the box and text.
    :param box_thickness: The border thickness of the box.
    :returns:             The image with the drawn box and text.
    """

    image = draw_rectangle(
        image,
        box_rect,
        color,
        box_thickness
    )

    text_x = box_rect[0] - box_thickness
    text_y = box_rect[1] - box_thickness

    image = draw_text(
        image,
        text,
        (text_x, text_y),
        color
    )

    return image
