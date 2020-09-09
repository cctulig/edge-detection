# Auto Tester
### app.py
Run app.py to run the auto tester.
This file loads the canvas that all the images and text are displayed on.
How to use the auto tester:
    - 'a' to accept an image crop
    - 'd' to deny an image crop
    - 'right arrow' to navigate to the next image
    - 'left arrow' to navigate to a previous image

### Scan Controller
Class that holds all the information of all the scans results

### Helper Functions
Functions for resizing the images as well as running the edge detection algorithms
IMPORTANT: Change line 32 to change which edge detection algorithm is being used.

# Edge Detection Algorithms:
## Top Performing Algorithm:
### Image Segmentation v2
1) Samples pixels in the center of the image using LAB color space
2) Masks pixels based on the color ranges determined from the sampling
3) Remove all but the smallest contour
4) Apply erosion and dilation
5) Find the contour of the screen
6) Draw the cropping rectangle

## Other Algorithms

### Image Segmentation v1
Unrefined version of v2, still uses LAB color space

### Image Segmentation v3
Same as v2, however uses the BGR color space instead of LAB with some different masking bounds.
I feel using this color space (maybe also in combination with hue), could be successful with more tuning.
Has the advantage of being a recognizable color space.

### Image Segmentation v4
Same as v2, however makes us of Canny to create rotated crops.

### Color Thresholding
Slightly adjusted algorithm written by Chris

### Canny Edge Detection
Uses OpenCV's Canny edge detection method in combination with clustering similar pixels together

### Canny Square Detection
Slightly adjusted algorithm written by Chris


