from tkinter import Canvas, NW

from PIL import Image, ImageTk

from canny_edge_detection import canny_edge_detection
from canny_square_detection import canny_square_detection
from color_thresholding import color_thresholding
from image_segmentation import image_segmentation
from image_segmentation_v2 import image_segmentation_v2
from image_segmentation_v3 import image_segmentation_v3
from image_segmentation_v4 import image_segmentation_v4
from random_scan import random_scan


def resize_image(im: Image, max: int):
    if im.width > im.height:
        prev_width = im.width
        height = int(im.height * max / prev_width)
        im2 = im.resize((max, height))
        return im2
    else:
        prev_height = im.height
        width = int(im.width * max / prev_height)
        im2 = im.resize((width, max))
        return im2


def open_next_image(canvas: Canvas, new_image):
    canvas.delete('scan')

    im = Image.open(new_image)
    scanned_imgs = image_segmentation_v2(im)

    im1 = open_image(canvas, scanned_imgs[0], 0, 0, 600)
    im2 = open_image(canvas, scanned_imgs[1], 650, 0, 300)
    im3 = open_image(canvas, scanned_imgs[2], 650, 300, 300)
    im4 = open_image(canvas, scanned_imgs[3], 950, 0, 300)
    im5 = open_image(canvas, scanned_imgs[4], 950, 300, 300)
    return [im1, im2, im3, im4, im5]


def open_image(canvas: Canvas, image: Image, x, y, size):
    image_resized = resize_image(image, size)
    photo_image = ImageTk.PhotoImage(image_resized)
    canvas.create_image(x, y, image=photo_image, anchor=NW, tags='scan')
    return photo_image


def fill_result_bar(result: int):
    if result == 1:
        return 'green'
    elif result == 2:
        return 'red'
    else:
        return 'grey'

