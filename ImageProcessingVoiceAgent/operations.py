import cv2
import numpy as np
from scipy import signal

""" OPERATIONS """
# Shift operation function
def operation_shift(img: np.ndarray , offset: int, direction: chr) -> np.ndarray:
    """ Shifts images to one of the four directions for a given number of pixels.

    Args:
        img (np.ndarray): Input image
        offset (int): Number of pixels to shift
        direction (chr): Which direction to shift
            's' -> South
            'w' -> West
            'n' -> North
            'e' -> East

    Returns:
        img_ (np.ndarray): Output image
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")

    if direction == 's':
        img_ = np.roll(img, shift=offset, axis=0)

    elif direction == 'w':
        img_ = np.roll(img, shift=offset, axis=1)

    elif direction == 'n':
        img = np.flip(img, 0)
        img = np.roll(img, shift=offset, axis=0)
        img_ = np.flip(img, 0)

    elif direction == 'e':
        img = np.flip(img, 1)
        img = np.roll(img, shift=offset, axis=1)
        img_ = np.flip(img, 1)

    else:
        raise Exception("Invalid axis in shift operation. Must be 'n', 'w', 's', or 'e'.")

    return img_

# Flip operation function
def operation_flip(img: np.ndarray, axis: chr) -> np.ndarray:
    """Flips the image in an axis

    Args:
        img (np.ndarray): Image input.
        direction (chr): Specify vertical or horizontal axis.

    Returns:
        img_ (np.ndarray): Output image
    """

    if axis == 'v':
        img_ = np.flip(img, 0)
    elif axis == 'h':
        img_ = np.flip(img, 1)
    else:
        raise Exception("Invalid axis. Must be 'v' or 'h'.")

    return img_

# Inverse operation function
def operation_inverse(img: np.ndarray) -> np.ndarray:
    """Returns the inverse of an image

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Inversed image
    """
    # This really takes the bitwise-NOT of the image!
    # But we can still use np.invert() or np.bitwise_not()
    img_ = ~img

    return img_

# Return image to grayscale
def operation_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts image to grayscale

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Grayscale image
    """    
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_

# Return the selected channel of the image either in color or grayscale 
def operation_channels(img: np.ndarray, channel: chr, colored: bool) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Define the channels
    blue, green, red = cv2.split(img)

    # Define channel having all zeros
    zeros = np.zeros(blue.shape, np.uint8)

    # merge zeros to make BGR image
    if channel == 'r':
        if colored:
            img_ = cv2.merge([zeros, zeros, red])
        else:
            img_ = red
    
    elif channel == 'b':
        if colored:
            img_ = cv2.merge([blue, zeros, zeros])
        else:
            img_ = blue

    elif channel == 'g':
        if colored:
            img_ = cv2.merge([zeros, green, zeros])
        else:
            img_ = green

    else: 
        raise Exception("Invalid channel. Must be 'r', 'g', or 'b'.")

    return img_

def operation_gaussian_blur(img: np.ndarray, amplitude_kernel: tuple[int, int]) -> np.ndarray:
    # Tuple elements must be unsigned odd integers
    # Kernel indexes can indicate directions. For example:
    # (3, 13) -> Vertical Blur
    # (13, 3) -> Horizontal Blur
    # (13, 13) -> Equal Blur
    if len(amplitude_kernel) != 2:
        raise ValueError("amplitude_kernel must contain 2 integers")
    if any((not isinstance(k, int) or k <= 0 or k % 2 == 0) for k in amplitude_kernel):
        raise ValueError("Kernel values must be positive odd integers")

    img_ = cv2.GaussianBlur(img, amplitude_kernel, 0)
    return img_

def operation_laplacian_sharpener(img: np.ndarray, 
                                  amplitude: float = 1,
                                  kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])) -> np.ndarray:
    # An example kernel:
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel = np.array(kernel, dtype=np.float32, copy=True)
    kernel *= amplitude
    
    img_ = cv2.filter2D(img, -1, kernel)
    return img_

def operation_convolve(img: np.ndarray, kernel: np.ndarray):
    grad = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    
    img_ = (np.absolute(grad), np.angle(grad))

    return img_


def operation_detect_edges(img: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Detects edges using Canny."""
    gray = operation_grayscale(img) if img.ndim == 3 else img
    return cv2.Canny(gray, threshold1, threshold2)


if __name__ == "__main__":
    demo_img = cv2.resize(cv2.imread("lenna.jpeg"), (0, 0), fx=0.5, fy=0.5)
    if demo_img is None:
        raise FileNotFoundError("lenna.jpeg was not found in the working directory")

    # Resize the image to make processing faster (optional)
    demo_img = cv2.resize(demo_img, (0, 0), fx=0.5, fy=0.5)

    transformed = operation_laplacian_sharpener(demo_img, 1)
    cv2.imshow("transformed_lenna", transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
shift+

horizontal flip+
vertical flip+

inverse+

grayscale+
RGB channel+

blur+
sharpen+

convolute+

derive
integrate

fourier
inverse fourier
"""