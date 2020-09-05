import cv2 as cv
import numpy as np
from PIL import Image
from cv2.cv2 import UMat
from matplotlib import pyplot as plt

from canny_square_detection import square_detection, find_largest_square


def image_segmentation_v4(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2LAB)

    height, width, _ = image.shape
    height_range = (height * .25, height * .75)
    width_range = (width * .25, width * .75)
    largest_area = 0
    largest_dim = [0, 0, 0, 0]

    tolerance = .3

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

    channel1_min += channel1_dif
    channel1_max += channel1_dif * 3
    channel2_min -= channel2_dif
    channel2_max += int(channel2_dif / 2)
    channel3_min -= channel3_dif
    channel3_max += int(channel3_dif / 2)

    lower_bound = np.array((channel1_min, channel2_min, channel3_min))
    upper_bound = np.array((channel1_max, channel2_max, channel3_max))

    mask = cv.inRange(image, lower_bound, upper_bound)

    out1 = find_largest_contour(mask)

    # erode + dilate
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(out1, kernel, iterations=2)
    out2 = find_largest_contour(erosion)
    dilate = cv.dilate(out2, kernel, iterations=2)

    converted = cv.cvtColor(image, cv.COLOR_LAB2BGR)
    screen = cv.bitwise_and(converted, converted, mask=dilate)


    squares = square_detection(dilate)

    # find contours, check for squares and then draw them
    if len(squares) > 0:
        large = find_largest_square(squares)
        cv.drawContours(converted, [large], -1, (36, 255, 12), 3)
    else:
        cnts, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv.contourArea)
        largest_dim = cv.boundingRect(cnt)

        cv.rectangle(converted, (largest_dim[0], largest_dim[1]),
                     (largest_dim[0] + largest_dim[2], largest_dim[1] + largest_dim[3]), (36, 255, 12), 3)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(converted, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(screen, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(out2, cv.COLOR_BGR2RGB))
    im4 = Image.fromarray(cv.cvtColor(erosion, cv.COLOR_BGR2RGB))
    im5 = Image.fromarray(cv.cvtColor(out1, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im4, im5]


def find_largest_contour(image):
    # Find largest contour in intermediate image
    cnts, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = max(cnts, key=cv.contourArea)

        # Output
        out = np.zeros(image.shape, np.uint8)
        cv.drawContours(out, [cnt], -1, 255, cv.FILLED)

        return out
    else:
        return image

