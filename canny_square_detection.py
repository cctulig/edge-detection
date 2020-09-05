import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
import argparse
from glob import glob


def canny_square_detection(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    # create 3 versions of the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv.filter2D(blur, -1, sharpen_kernel)

    squares = square_detection(sharpen)
    final = image.copy()

    # check for squares and then draw them
    if len(squares) > 0:
        large = find_largest_square(squares)
        cv.drawContours(final, squares, -1, (0, 255, 0), 3)
        cv.drawContours(final, [large], -1, (0, 0, 255), 3)
    else:
        cv.drawContours(final, squares, -1, (0, 255, 0), 3)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(final, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(sharpen, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(blur, cv.COLOR_BGR2RGB))
    im4 = Image.fromarray(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
    im5 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im4, im5, None]


# Helper functions
def angle_cos(p0, p1, p2):
    """
        Calculates the cosine
    """
    d1, d2 = (p0 - p1).astype(float), (p2 - p1).astype(float)
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def area(XY, n):
    """
        Calculates the area of a polygon with n vertices using the shoelace formula
    """
    area = 0.0
    j = n - 1  # j is previous vertex to i
    for i in range(0, n):
        area += (XY[j][0] + XY[i][0]) * (XY[j][1] - XY[i][1])
        j = i
    return int(abs(area / 2.0))


def find_largest_square(sqrs):
    """
        Finds the larges square based upon area
    """
    res = sqrs[0]
    for i in sqrs:
        if area(i, 4) > area(res, 4):
            res = i
    return res


def square_detection(img):
    # smooth the image
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []

    # split the images channels
    for gray in cv.split(img):
        # loop over thresholds
        for thrs in range(0, 255, 26):
            # set the thre
            if thrs == 0:
                bins = cv.Canny(gray, 0, 50, apertureSize=5)
                bins = cv.dilate(bins, None)
            else:
                _retval, bins = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)

            contours, _hierarchy = cv.findContours(bins, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            squares = find_squares(contours)
    return squares


def find_squares(contours):
    squares = []
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    return squares

