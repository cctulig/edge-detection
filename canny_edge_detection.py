import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
import argparse
from glob import glob

from canny_square_detection import square_detection, find_largest_square, find_squares
from color_thresholding import find_largest_contour


def canny_edge_detection(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    # Clustering
    # cluster2 = clustering(image, 2)
    # cluster4 = clustering(image, 4)
    # cluster8 = clustering(image, 8)
    # cluster16 = clustering(image, 16)

    # Manipulate Image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv.filter2D(blur, -1, sharpen_kernel)
    edges = cv.Canny(image, 50, 100, apertureSize=3, L2gradient=True)

    # dilate to combine adjacent text contours
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(edges, kernel, iterations=2)

    # Find Contours
    contours, _hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    squares = find_squares(contours)
    final = image.copy()

    # Find Contours v2
    copy = image.copy()
    largest_dim = find_largest_contour(contours, image)
    cv.rectangle(copy, (largest_dim[0], largest_dim[1]), (largest_dim[0] + largest_dim[2], largest_dim[1] + largest_dim[3]), (36, 255, 12), 3)
    print(largest_dim)

    # check for squares and then draw them
    if len(squares) > 0:
        large = find_largest_square(squares)
        cv.drawContours(final, squares, -1, (0, 255, 0), 3)
        cv.drawContours(final, [large], -1, (0, 0, 255), 3)
    else:
        cv.drawContours(final, squares, -1, (0, 255, 0), 3)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(final, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(copy, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(dilate, cv.COLOR_BGR2RGB))
    im4 = Image.fromarray(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
    im5 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im4, im5, None]


def clustering(img, K):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    return res2

