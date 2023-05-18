import os
import cv2
import numpy as np
from math import sqrt
from skimage import io, color, exposure, util
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def measure_pigment_network(image): # Feature 1
    
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the L, A, and B channels from the LAB image
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply contrast stretching to enhance the L channel
    enhanced_l_channel = cv2.equalizeHist(l_channel)

    # Perform thresholding on the enhanced L channel to obtain a binary mask
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise and refine the binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphological_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Convert the morphological mask to a color image
    color_mask = cv2.cvtColor(morphological_mask, cv2.COLOR_GRAY2BGR)

    # Combine the color mask with the original image to highlight the regions of interest
    result = cv2.bitwise_and(image, color_mask)

    # Calculate the percentage of pigment network coverage
    total_pixels = np.prod(binary_mask.shape[:2])
    pigment_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    if coverage_percentage > 50:
        return 1
    else:
        return 0


def measure_blue_veil(image): # Feature 2
    height_picture, width_picture, _ = image.shape
    total_pixels = height_picture * width_picture
    count = 0

    for y in range(height_picture):
        for x in range(width_picture):
            b_picture = float(image[y, x][0])  # Blue channel value
            g_picture = float(image[y, x][1])  # Green channel value
            r_picture = float(image[y, x][2])  # Red channel value

            total = r_picture + g_picture + b_picture

            if b_picture > 60 and (r_picture - 46 < g_picture) and (g_picture < r_picture + 15):
                count += 1

    if count > 0:
        return 1
    else:
        return 0


def measure_vascular(image): # Feature 3
    # Enhance the red color
    red_channel = image[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1)
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
        return 1
    else:
        return 0


def measure_globules(image): # Feature 4
    # Preprocess the image
    image_gray = rgb2gray(image)
    inverted_image = util.invert(image_gray)

    # Detect blobs
    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)
    if blob_amount > 600:
        return 1
    else:
        return 0


def measure_streaks(image): # Feature 5
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute area of the skin lesion
    lesion_area = cv2.contourArea(contours[0])

    # Compute perimeter of the border
    border_perimeter = cv2.arcLength(contours[0], True)

    irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    threshold = 1.8
    if irregularity > threshold:
        return 1
    else:
        return 0


def measure_irregular_pigmentation(image): # Feature 6
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a binary image
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    
    # Label connected components in the binary image
    labeled_image = label(binary)
    
    # Initialize lists to store irregular pigmentation regions' coordinates
    min_rows, min_cols, max_rows, max_cols = [], [], [], []
    
    # Iterate through each labeled region
    for region in regionprops(labeled_image):
        # Calculate the area and perimeter of the region
        area = region.area
        perimeter = region.perimeter
        
        # Calculate the circularity of the region
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # Check if the region is irregular based on circularity threshold
        if circularity < 0.6:
            # Get the bounding box coordinates of the region
            min_row, min_col, max_row, max_col = region.bbox
            
            # Store the coordinates of the irregular pigmentation region
            min_rows.append(min_row)
            min_cols.append(min_col)
            max_rows.append(max_row)
            max_cols.append(max_col)
    
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the L channel from the LAB image
    l_channel = lab_image[:, :, 0]

    # Apply adaptive thresholding to create a binary mask
    _, binary_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the percentage of irregular pigmentation coverage
    total_pixels = np.prod(binary_mask.shape[:2])
    irregular_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (irregular_pixels / total_pixels) * 100

    if coverage_percentage > 50:
        return 1
    else:
        return 0


def measure_regression(image): # Feature 7
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the scar-like white/blue color
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])

    # Create a mask using the defined color bounds
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Count the number of non-zero pixels in the mask
    num_pixels = cv2.countNonZero(mask)

    threshold = 2500
    if num_pixels > threshold:
        return 1
    else:
        return 0


# Test the functions with example images
image_path = 'path_to_image.jpg'
image = cv2.imread(image_path)

print("Pigment Network:", measure_pigment_network(image))
print("Blue Veil:", measure_blue_veil(image))
print("Vascular Structures:", measure_vascular(image))
print("Globules:", measure_globules(image))
print("Streaks:", measure_streaks(image))
print("Irregular Pigmentation:", measure_irregular_pigmentation(image))
print("Regression Structures:", measure_regression(image))