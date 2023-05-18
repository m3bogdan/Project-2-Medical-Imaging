#used this to detected the irregular edges
#https://www.sciencedirect.com/science/article/abs/pii/089561119290074J?via%3Dihub

import cv2
import numpy as np

# Load image
img = cv2.imread('/Users/bogdancristianmihaila/Desktop/2nd Semester/Github/project2/Project-2-Medical-Imaging/data/images/test_foto_res/mask_resized_point5_2.png')

# Convert to grayscale and apply threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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