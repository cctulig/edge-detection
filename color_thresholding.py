import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def color_thresholding(pil_img: Image):
    # Convert PIL image to OpenCV image
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    height, width, _ = image.shape
    height_range = (height * .25, height * .75)
    width_range = (width * .25, width * .75)
    largest_area = 0
    largest_dim = [0, 0, 0, 0]

    # image = histogram_equalization(image)

    # # edge detection filter
    # kernel = np.array([[0.0, -2.0, 0.0],
    #                    [-2.0, 12.0, -2.0],
    #                    [0.0, -2.0, 0.0]])
    #
    # kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    #
    # # filter the source image
    # img_rst = cv.filter2D(image, -1, kernel)

    # apply grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # remove glare
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # apply blur
    blur = cv.GaussianBlur(gray, (9, 9), 0)

    # apply  threshold
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # dilate to combine adjacent text contours
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find contours, highlight text areas, and extract ROIs
    cntss = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cntss[0] if len(cntss) == 2 else cntss[1]

    for c in cnts:
        area = cv.contourArea(c)
        if area > 10000:
            print('image c area: {0}'.format(area))
            x, y, w, h = cv.boundingRect(c)
            print('x: {0}'.format(x))
            if width_range[0] < x + w / 2 < width_range[1] and height_range[0] < y + h / 2 < height_range[1] and w * h > largest_area:
                largest_area = w * h
                largest_dim = [x, y, w, h]

    cv.rectangle(image, (largest_dim[0], largest_dim[1]), (largest_dim[0] + largest_dim[2], largest_dim[1] + largest_dim[3]), (36, 255, 12), 3)

    for c in cnts:
        area = cv.contourArea(c)
        if area > 10000:
            print('dilate c area: {0}'.format(area))
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(dilate, (x, y), (x + w, y + h), (200, 200, 200), 20)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(dilate, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(thresh, cv.COLOR_BGR2RGB))
    im4 = Image.fromarray(cv.cvtColor(blur, cv.COLOR_BGR2RGB))
    im5 = Image.fromarray(cv.cvtColor(equalized, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im4, im5, largest_dim]


# function for color image equalization
def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv.equalizeHist(b)
    equ_g = cv.equalizeHist(g)
    equ_r = cv.equalizeHist(r)
    equ = cv.merge((equ_b, equ_g, equ_r))
    # print(equ)
    # cv.imwrite('output_name.png', equ)
    return img_out


def find_largest_contour(cnts, image):
    height, width, _ = image.shape
    height_range = (height * .25, height * .75)
    width_range = (width * .25, width * .75)
    largest_area = 0
    largest_dim = [0, 0, 0, 0]

    for c in cnts:
        area = cv.contourArea(c)
        if area > 10000:
            x, y, w, h = cv.boundingRect(c)
            if width_range[0] < x + w / 2 < width_range[1] and height_range[0] < y + h / 2 < height_range[1] and w * h > largest_area:
                largest_area = w * h
                largest_dim = [x, y, w, h]

    return largest_dim

