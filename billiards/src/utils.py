import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import re

GREEN = (0, 255, 0)


def listdir(path, ignore_folders=False):
    files = list(filter(lambda x: x.startswith('.') is False, os.listdir(path)))
    if ignore_folders:
        files = list(filter(lambda x: '.' in x, files))
    return files


def energy_level(string):
    result = re.search(r'n=\d*_', string)
    energy_level = int(result.group(0).strip('n=_'))
    return energy_level


def white_portion(working_piece):
    piece_height, piece_width = working_piece.size
    image_array = np.array(working_piece)

    num_white_pixels = (image_array > 200).sum()

    return num_white_pixels / (piece_height * piece_width)


def apply_max_contour(img, cnts, debug=False):
    max_contour = max(cnts, key=cv2.contourArea)

    mask = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(mask, [max_contour], [255, 255, 255])

    # Apply mask.
    masked_img = cv2.bitwise_and(img, mask)
    if debug:
        plt.imshow(masked_img)
        plt.show()

    # Green background.
    background = np.full(masked_img.shape, GREEN, dtype=np.uint8)
    mask_not = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, mask_not)

    # Sum `masked_img` with `background`.
    result = background + masked_img
    x, y, w, h = cv2.boundingRect(max_contour)
    result = result[y:y + h, x:x + w]
    return result


def colors_from_array(img_array):
    result_img = Image.fromarray(img_array)
    clrs = result_img.getcolors(1000)
    tmp = dict(clrs)
    clrs = dict(zip(tmp.values(), tmp.keys()))
    width, height = result_img.size
    return clrs, width, height


def find_contours(img, minw, minh, debug=False):
    def isnt_small(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return w > minw and h > minh

    # To gray scale.
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize.
    _, gray_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    if debug:
        plt.imshow(gray_image)
        plt.show()

    # Find contours.
    edged = cv2.Canny(gray_image, 10, 250)
    if debug:
        plt.imshow(edged)
        plt.show()

    # Largest contour.
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(filter(lambda x: isnt_small(x), cnts))
    return cnts


def random_piece(image, piece_width, piece_height):
    width, height = image.size
    up = np.random.randint(0, height - piece_height)
    down = up + piece_height
    left = np.random.randint(0, width - piece_width)
    right = left + piece_width

    piece = image.crop((left, up, right, down))
    return piece


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
