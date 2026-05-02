import PIL
import cv2
import numpy as np

img = cv2.imread("lenna.jpeg")

def operation_shift(img: np.ndarray , offset: int, direction: chr) -> np.ndarray:
    """ Shifts images to one of the four directions for a given number of pixels.
    's' -> South
    'w' -> West
    'n' -> North
    'e' -> East

    Args:
        img (np.ndarray): Image input
        offset (int): Number of pixels to shift
        direction (chr): Which direction to shift

    Returns:
        np.ndarray: Returns the transformed image.
    """

    if direction == 's':
        img_ = np.roll(img, shift= offset, axis= 0)

    elif direction == 'w':
        img_ = np.roll(img, shift= offset, axis= 1)

    elif direction == 'n':
        img = np.flip(img, 0)
        img = np.roll(img, shift= offset, axis= 0)
        img_ = np.flip(img, 0)

    elif direction == 'e':
        img = np.flip(img, 1)
        img = np.roll(img, shift= offset, axis= 1)
        img_ = np.flip(img, 1)

    else:
        raise Exception("Invalid axis in shift operation.")

    return img_

cv2.imshow("lenna", img)
cv2.waitKey(0)

img = operation_shift(img, 90, 'e')
cv2.imshow("lenna", img)
cv2.waitKey(0)

cv2.destroyAllWindows()

"""
shift

horizontal flip 
vertical flip

inverse

monochrome
RGB channel

blur
sharpen

convolute

derive
integrate

fourier
inverse fourier
"""