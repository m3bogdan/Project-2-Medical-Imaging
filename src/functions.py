from skimage import io, color
import numpy as np
import os
from skimage import exposure
from skimage.color import rgb2gray
from math import sqrt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage import util
import cv2


def measure_vascular(image): #feature 3

    # Enhance the red colour (idk if it does much)
    red_channel = image[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1) #gamma was 1.5 before, changed bcs of error
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel

    hsv_img = color.rgb2hsv(enhanced_image) 

    lower_red = np.array([0/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red = np.array([25/360, 1, 1])  # Upper limit for red hue, saturation, and value

    lower_red2 = np.array([330/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red2 = np.array([1, 1, 1])  # Upper limit for red hue, saturation, and value

    mask = np.logical_or(
        np.logical_and(np.all(hsv_img >= lower_red, axis=-1), np.all(hsv_img <= upper_red, axis=-1)),
        np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1)))

    result = enhanced_image.copy()
    result[~mask] = 0
    
    if result.max() > 0:
        print("Diagnosis: cancer, vascular pattern found")
    else:
        print("Diagnosis: not a cancer")
        
def measure_globules(image): #feature 4
    # Preprocess the image
    image_gray = rgb2gray(image)
    inverted_image = util.invert(image_gray)

    # Detect blobs
    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05) #min_sigma was 0.8 changed bcs of error
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)
    if blob_amount > 600:
        print("Diagnosis: cancer, blobs found: ", blob_amount)
    else:
        print("Diagnosis: not a cancer, blobs found: ", blob_amount)
        
def measure_streaks(image): #feature 5
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute area of the skin lesion
    lesion_area = cv2.contourArea(contours[0])

    # Compute perimeter of the border
    border_perimeter = cv2.arcLength(contours[0], True)

    print('Lesion area:', lesion_area)
    print('Border perimeter:', border_perimeter)

    irregularity = (border_perimeter ** 2) / 4 * np.pi * lesion_area
    print('Irregularity:', irregularity)
    
    threshold = 1.8
    if irregularity > threshold:
        print('Irregular streaks detected!')
    else:
        print('No irregular streaks detected.')
        
def measure_regression(image): #feature 7
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the scar-like white/blue color
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])

    # Create a mask using the defined color bounds
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Apply the mask to the image
    #masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Count the number of non-zero pixels in the mask
    num_pixels = cv2.countNonZero(mask)
    print(num_pixels)

    # Check if the number of non-zero pixels is above a threshold
    threshold = 2500 # more fine tunning needed(aka more images to test on)
    if num_pixels > threshold:
        print('1')
    else:
        print('0')