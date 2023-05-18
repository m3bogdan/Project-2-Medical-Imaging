import os
from skimage import io
import cv2
import numpy as np



def blue_veil_detect(picture):
    height_picture, width_picture, _ = picture.shape
    total_pixels = height_picture * width_picture
    count = 0

    for y in range(height_picture):
        for x in range(width_picture):
            b_picture = float(picture[y, x][0])  # Blue channel value
            g_picture = float(picture[y, x][1])  # Green channel value
            r_picture = float(picture[y, x][2])  # Red channel value

            total = r_picture + g_picture + b_picture

            if b_picture > 60 and (r_picture - 46 < g_picture) and (g_picture < r_picture + 15):
                count += 1

    if count > 0:
        #return round(count/total_pixels * 100,2)
        #return count
        return 1
    else:
        return 0
#a
#print(blue_veil_detect(masked_image))
