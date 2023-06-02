import cv2
import numpy as np

# Load the image
img = cv2.imread('/Users/bogdancristianmihaila/Desktop/2nd Semester/Github/project2/Project-2-Medical-Imaging/data/images/test_photos/point7_cmask.png')

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
