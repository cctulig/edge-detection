import cv2 as cv
import numpy as np
from PIL import Image
from cv2.cv2 import UMat
from matplotlib import pyplot as plt

from color_thresholding import color_thresholding, histogram_equalization


def image_segmentation(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2LAB)

    height, width, _ = image.shape
    height_range = (height * .25, height * .75)
    width_range = (width * .25, width * .75)
    largest_area = 0
    largest_dim = [0, 0, 0, 0]

    tolerance = .15

    channel1_min = 255
    channel1_max = -255
    channel2_min = 255
    channel2_max = -255
    channel3_min = 255
    channel3_max = -255

    x_max = int(width * .54)
    y_max = int(height * .54)
    x = int(width * .46)
    while x < x_max:
        y = int(height * .46)
        while y < y_max:
            pixel = image[y, x]
            channel1_min = min(channel1_min, pixel[0])
            channel1_max = max(channel1_max, pixel[0])
            channel2_min = min(channel2_min, pixel[1])
            channel2_max = max(channel2_max, pixel[1])
            channel3_min = min(channel3_min, pixel[2])
            channel3_max = max(channel3_max, pixel[2])
            y += 1
        x += 1

    channel1_dif = int((channel1_max - channel1_min) * tolerance)
    channel2_dif = int((channel2_max - channel2_min) * tolerance)
    channel3_dif = int((channel3_max - channel3_min) * tolerance)

    channel1_min -= channel1_dif
    channel1_max += channel1_dif
    channel2_min -= channel2_dif
    channel2_max += channel2_dif
    channel3_min -= channel3_dif
    channel3_max += channel3_dif

    lower_bound = np.array((channel1_min, channel2_min, channel3_min))
    upper_bound = np.array((channel1_max, channel2_max, channel3_max))

    mask = cv.inRange(image, lower_bound, upper_bound)
    result = cv.bitwise_and(image, image, mask=mask)
    im = color_thresholding(Image.fromarray(cv.cvtColor(result, cv.COLOR_LAB2RGB)))

    # mask = 255 - mask
    # result = cv.bitwise_and(image, image, mask=mask)
    # im3 = color_thresholding(Image.fromarray(cv.cvtColor(result, cv.COLOR_LAB2RGB)))[0]

    largest_dim = im[5]
    converted = cv.cvtColor(image, cv.COLOR_LAB2BGR)
    cv.rectangle(converted, (largest_dim[0], largest_dim[1]), (largest_dim[0] + largest_dim[2], largest_dim[1] + largest_dim[3]), (36, 255, 12), 3)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(converted, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(result, cv.COLOR_LAB2RGB))
    # im3 = Image.fromarray(cv.cvtColor(image, cv.COLOR_HSV2RGB))
    # im4 = Image.fromarray(cv.cvtColor(image, cv.COLOR_LAB2RGB))
    # im5 = Image.fromarray(cv.cvtColor(image, cv.COLOR_LAB2RGB))

    return [im1, im[1], im[2], im[3], im2]


