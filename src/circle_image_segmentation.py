# import the necessary packages
import matplotlib.pyplot as plt
from skimage import color, exposure, io, morphology
import cv2
import numpy as np
import os

def step1(original_image):
    # load image
    img = cv2.imread(original_image)
    # convert image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # calculate center coordinates and radius of circular mask
    (h, w) = img.shape[:2]
    (center_x, center_y) = (w // 2, h // 2)
    radius = int(min(h, w) * 0.4)  # set radius as 40% of minimum image dimension
    # create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, color=1, thickness=-1)
    # apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img, img 


def step2(masked_img):
    # Convert the image to grayscale
    image_gray = color.rgb2gray(masked_img)
    # Create a binary mask
    mask = image_gray < 0.4
    # Apply the mask to the image
    image_masked = masked_img.copy()
    image_masked[~mask] = 0
    # Apply erosion
    image_eroded = morphology.binary_erosion(mask)
    # Apply dilation
    image_dilated = morphology.binary_dilation(image_eroded)
    mask_dilated = morphology.binary_dilation(image_dilated, morphology.disk(6))
    #put the image in the mask
    image_masked = masked_img.copy()
    image_masked[~mask_dilated] = 0
    return image_masked


def step3(image_masked,img):
    # calculate center coordinates and radius of circular mask
    (h, w) = img.shape[:2]
    (center_x, center_y) = (w // 2, h // 2)
    radius = int(min(h, w) * 0.35)  # set radius as 40% of minimum image dimension
    # create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, color=255, thickness=-1)
    # apply mask to image
    masked2_img = cv2.bitwise_and(image_masked, image_masked, mask=mask)
    #make the colours normal again
    masked2_img = cv2.cvtColor(masked2_img, cv2.COLOR_BGR2RGB)
    return masked2_img


def image_mask(pathfolder, folder_path_out):
    #The saving of the images of step1 and step2 has been commented out,
    # if you want them, uncomment line 67 and 70 (or opposite)

    # for each images in the folder, we run the function
    for filename in os.listdir(pathfolder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            #print(os.path.join(pathfolder, filename))
            x,img=step1(os.path.join(pathfolder, filename))
            # save x with the same name +step1
            cv2.imwrite(os.path.join(folder_path_out, filename[:-4]+'_step1.jpg'), x)
            y = step2(x)
            # save y with the same name +step2
            cv2.imwrite(os.path.join(folder_path_out, filename[:-4]+'_step2.jpg'), y)
            z = step3(y,img)
            # save z with the same name + step3 (_masked becasue it's a final result)
            cv2.imwrite(os.path.join(folder_path_out, filename[:-4]+'_masked.jpg'), z)        
            continue
        else:
            continue

#put the path of the folder where you want to save the image
pathfolder = "C:/Users/annam/Desktop/Globules/Resized"
folder_path_out = "C:/Users/annam/Desktop/Globules/Segmentation"

image_mask(pathfolder, folder_path_out)