import os
import cv2


def measure_blue_veil(image):
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
    return count




# Directories containing the pictures
directory1 = "C:/Users/annam/Desktop/cancer"
directory2 = "C:/Users/annam/Desktop/no_cancer"

# Lists to store the results
cancer = []
no_cancer = []

# Function to process a directory
def process_directory(directory, results, measure_function):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a picture
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the picture
            picture_path = os.path.join(directory, filename)
            picture = cv2.imread(picture_path)

            # Process the picture using the provided measure_function
            result = measure_function(picture)

            # Append the result to the list
            results.append(result)




measure_function = measure_blue_veil
# Process the first directory using measure_blue_veil function
process_directory(directory1, cancer, measure_function)

# Process the second directory using measure_blue_veil function
process_directory(directory2, no_cancer, measure_function)

# Print the results
print("Results from directory cancers:", cancer)
print("Results from directory with non_cancers:", no_cancer)

