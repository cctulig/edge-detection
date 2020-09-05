from random import randrange

from PIL import ImageDraw
from PIL.Image import Image


def random_scan(image: Image):
    width = image.width
    height = image.height
    side = int(min(width, height) / 2)
    xPos = randrange(0, width - side)
    yPos = randrange(0, height - side)
    img_draw = ImageDraw.Draw(image)
    img_draw.rectangle((xPos, yPos, xPos + side, yPos + side), outline='red', width=10)
    return [image] * 5

