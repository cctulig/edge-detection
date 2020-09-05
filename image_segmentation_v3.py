from numpy import std

import cv2 as cv
import numpy as np
from PIL import Image
from cv2.cv2 import UMat
from matplotlib import pyplot as plt

from image_segmentation_v2 import find_largest_contour


def image_segmentation_v3(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    height, width, _ = image.shape
    height_range = (height * .25, height * .75)
    width_range = (width * .25, width * .75)
    largest_area = 0
    largest_dim = [0, 0, 0, 0]

    tolerance = 30

    channel1_data = []
    channel2_data = []
    channel3_data = []

    x_max = int(width * .53)
    y_max = int(height * .53)
    x = int(width * .47)
    while x < x_max:
        y = int(height * .47)
        while y < y_max:
            pixel = image[y, x]
            channel1_data.append(pixel[0])
            channel2_data.append(pixel[1])
            channel3_data.append(pixel[2])
            y += 1
        x += 1

    channel1_stdv = std(channel1_data)
    channel2_stdv = std(channel2_data)
    channel3_stdv = std(channel3_data)

    channel1_min = channel1_stdv - tolerance
    channel2_min = channel2_stdv - tolerance
    channel3_min = channel3_stdv - tolerance

    channel1_max = channel1_stdv + tolerance
    channel2_max = channel2_stdv + tolerance
    channel3_max = channel3_stdv + tolerance

    lower_bound = np.array((channel1_min, channel2_min, channel3_min))
    upper_bound = np.array((channel1_max, channel2_max, channel3_max))

    print(lower_bound)
    print(upper_bound)

    mask = cv.inRange(image, lower_bound, upper_bound)

    out1 = find_largest_contour(mask)

    # erode + dilate
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(out1, kernel, iterations=2)
    out2 = find_largest_contour(erosion)

    dilate = cv.dilate(out2, kernel, iterations=2)

    screen = cv.bitwise_and(image, image, mask=dilate)

    # find contour
    cnts, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = max(cnts, key=cv.contourArea)
        largest_dim = cv.boundingRect(cnt)

        cv.rectangle(image, (largest_dim[0], largest_dim[1]), (largest_dim[0] + largest_dim[2], largest_dim[1] + largest_dim[3]), (36, 255, 12), 3)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(screen, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(out2, cv.COLOR_BGR2RGB))
    im4 = Image.fromarray(cv.cvtColor(erosion, cv.COLOR_BGR2RGB))
    im5 = Image.fromarray(cv.cvtColor(out1, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im4, im5]