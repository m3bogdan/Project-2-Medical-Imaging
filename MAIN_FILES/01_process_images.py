import os
import cv2
import numpy as np
import csv
from math import sqrt
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def combine_images(mask_folder, picture_folder, output_folder):
    """
    Combine masked images based on the provided masks and original pictures.

    Args:
        mask_folder (str): Path to the folder containing mask images.
        picture_folder (str): Path to the folder containing original picture images.
        output_folder (str): Path to the output folder to save the combined images.

    Returns:
        None
    """
    # Get a list of files in the mask folder
    mask_files = os.listdir(mask_folder)
    picture_files = os.listdir(picture_folder)

    # Iterate over each mask file
    for mask_file in mask_files:
        # Extract the file name without extension
        mask_filename = os.path.splitext(mask_file)[0]

        # Check if the corresponding picture file exists
        if mask_filename in picture_files:
            # Construct the full paths for mask and picture files
            mask_path = os.path.join(mask_folder, mask_file)
            picture_path = os.path.join(picture_folder, mask_filename)

            # Load the mask and picture images
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            picture_image = cv2.imread(picture_path)

            # Apply the mask to the picture
            masked_image = cv2.bitwise_and(picture_image, picture_image, mask=mask_image)

            # Generate the output file name
            output_file = mask_filename + '_masked.png'

            # Construct the full path for the output file
            output_path = os.path.join(output_folder, output_file)

            # Save the masked image to the output folder
            cv2.imwrite(output_path, masked_image)


def image_resize(folder_path_in):
    """
    Resize images in the input folder to a height of 256 pixels while preserving the aspect ratio.

    Args:
        folder_path_in (str): Path to the folder containing images to be resized.

    Returns:
        None
    """
    # Iterate through all the jpg and png files in the input folder
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        # Read in the image
        image_path = os.path.join(folder_path_in, filename)
        original = cv2.imread(image_path)

        # Ignore the alpha channel (e.g. transparency)
        if original.shape[-1] == 4:
            original = original[..., :3]

        # Resize the image (preserving the proportions)
        new_height = 256
        new_width = int(new_height / original.shape[0] * original.shape[1])
        resized = cv2.resize(original, (new_width, new_height))

        # Save the resized image, replacing the original file
        cv2.imwrite(image_path, resized)


def measure_pigment_network(image):
    """
    Measure the coverage percentage of the pigment network in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Coverage percentage of the pigment network.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    enhanced_l_channel = cv2.equalizeHist(l_channel)
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = np.prod(binary_mask.shape[:2])
    pigment_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    return coverage_percentage


def measure_blue_veil(image):
    """
    Measure the number of pixels exhibiting blue veil in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels with blue veil.
    """
    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    return count


def measure_vascular(image):
    """
    Measure the presence of vascular structures in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels representing vascular structures.
    """
    red_channel = image[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1)
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel
    hsv_img = color.rgb2hsv(enhanced_image)

    lower_red1 = np.array([0, 40/100, 00/100])
    upper_red1 = np.array([25/360, 1, 1])
    mask1 = np.logical_and(np.all(hsv_img >= lower_red1, axis=-1), np.all(hsv_img <= upper_red1, axis=-1))

    lower_red2 = np.array([330/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red2 = np.array([1, 1, 1])  # Upper limit for red hue, saturation, and value
    mask2 = np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1))

    mask = np.logical_or(mask1, mask2)

    return np.sum(mask)


def measure_globules(image):
    """
    Measure the number of globules in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of globules.
    """
    image_gray = rgb2gray(image)
    inverted_image = 1 - image_gray

    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)

    return blob_amount


def measure_streaks(image):
    """
    Measure the irregularity of streaks in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Irregularity measure of streaks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area = cv2.contourArea(contours[0])
    border_perimeter = cv2.arcLength(contours[0], True)
    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    return irregularity


def measure_irregular_pigmentation(image):
    """
    Measure the coverage percentage of irregular pigmentation in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Coverage percentage of irregular pigmentation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    labeled_image = label(binary)

    min_rows, min_cols, max_rows, max_cols = [], [], [], []

    for region in regionprops(labeled_image):
        area = region.area
        perimeter = region.perimeter

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity < 0.6:
            min_row, min_col, max_row, max_col = region.bbox
            min_rows.append(min_row)
            min_cols.append(min_col)
            max_rows.append(max_row)
            max_cols.append(max_col)

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_pixels = np.prod(binary_mask.shape[:2])
    irregular_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (irregular_pixels / total_pixels) * 100

    return coverage_percentage


def measure_regression(image):
    """
    Measure the number of pixels representing regression structures in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels representing regression structures.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    num_pixels = cv2.countNonZero(mask)

    return num_pixels


def extract_features(image_path):
    """
    Extract features from an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    image = cv2.imread(image_path)

    features = {}
    features['filename'] = os.path.basename(image_path)
    features['pigment_network_coverage'] = measure_pigment_network(image)
    features['blue_veil_pixels'] = measure_blue_veil(image)
    features['vascular_pixels'] = measure_vascular(image)
    features['globules_count'] = measure_globules(image)
    features['streaks_irregularity'] = measure_streaks(image)
    features['irregular_pigmentation_coverage'] = measure_irregular_pigmentation(image)
    features['regression_pixels'] = measure_regression(image)

    return features


def save_features_to_csv(features_list, output_file):
    """
    Save a list of features dictionaries to a CSV file.

    Args:
        features_list (list): List of features dictionaries.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    keys = features_list[0].keys()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features_list)


def main():
    # Set the paths for input folders
    mask_folder = r'MAIN_FILES\MAIN_DATA\Input\Mask_folder'
    picture_folder = r'MAIN_FILES\MAIN_DATA\Input\Picture_folder'
    output_folder = r'MAIN_FILES\MAIN_DATA\Input\Processed_folder'

    # Combine masked images
    combine_images(mask_folder, picture_folder, output_folder)

    # Resize images in the output folder
    image_resize(output_folder)

    # Set the path for the output CSV file
    output_file = r'MAIN_FILES\MAIN_DATA\Input\features.csv'

    # Initialize a list to store the extracted features
    features_list = []

    # Iterate through all the resized images in the output folder
    for filename in [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]:
        # Extract features from the image
        image_path = os.path.join(output_folder, filename)
        features = extract_features(image_path)

        # Append the features to the list
        features_list.append(features)

    # Save the features to a CSV file
    save_features_to_csv(features_list, output_file)


if __name__ == '__main__':
    main()
