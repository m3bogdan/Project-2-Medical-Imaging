import os
import cv2
import csv
import shutil
import extract_features as feature
from PIL import Image
import os


#########################################
###        Images preprocessing       ###
#########################################

def superpose_segmentation(normal_folder, segmentation_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all the files in the normal folder
    normal_files = os.listdir(normal_folder)

    for file in normal_files:
        # Check if the file has a corresponding segmentation image
        segmentation_file = os.path.join(segmentation_folder, file.replace('.jpg', '_segmentation.png'))
        if os.path.exists(segmentation_file):
            try:
                # Open the normal image and the segmentation image
                normal_image = Image.open(os.path.join(normal_folder, file)).convert("RGBA")
                segmentation_image = Image.open(segmentation_file).convert("RGBA")

                # Resize the images to 256x256 pixels
                normal_image = normal_image.resize((256, 256))
                segmentation_image = segmentation_image.resize((256, 256))

                # Invert the segmentation image
                inverted_segmentation = Image.eval(segmentation_image, lambda x: 255 - x)

                # Create a binary mask from the inverted segmentation image
                mask = inverted_segmentation.split()[0].point(lambda x: 255 if x == 0 else 0).convert("L")

                # Apply the mask to the normal image
                normal_image.putalpha(mask)

                # Create a black background
                background = Image.new("RGBA", normal_image.size, (0, 0, 0, 255))
                # Composite the normal image with the black background
                result = Image.alpha_composite(background, normal_image)

                # Save the superposed image to the output folder
                output_file = os.path.join(output_folder, file)
                result.save(output_file, "PNG")

                print(f"Superposed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        else:
            print(f"No segmentation image found for {file}")

#########################################
###         Feature extraction        ###
#########################################

def extracting_features(image_path):
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
    features['pigment_network_coverage'] = feature.measure_pigment_network(image)
    features['blue_veil_pixels'] = feature.measure_blue_veil(image)
    features['vascular_pixels'] = feature.measure_vascular(image)
    features['globules_count'] = feature.measure_globules(image)
    features['streaks_irregularity'] = feature.measure_streaks(image)
    features['irregular_pigmentation_coverage'] = feature.measure_irregular_pigmentation(image)
    features['regression_pixels'] = feature.measure_regression(image)

    return features


def save_features_to_csv(features_list, output_file):
    """
    Save a list of features dictionaries to a CSV file.

    Args:
        features_list (list): List of dictionaries containing image features.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    if not features_list:
        print("No features found.")
        return

    keys = features_list[0].keys()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features_list)

    print(f"Features saved to {output_file}.")


def main():

    # Provide the paths to the folders containing the images
    normal_folder = r"C:\Users\serru\Downloads\img\Test"
    segmentation_folder = r"C:\Users\serru\Downloads\img\Binary_mask\Test"
    output_folder = r"C:\Users\serru\Downloads\img\Output"

    #Pre-process the images
    superpose_segmentation(normal_folder, segmentation_folder, output_folder)


    # Set the path for the output CSV file
    output_file = r'C:\Users\serru\Downloads\img\features.csv'

    # Initialize a list to store the extracted features
    features_list = []

    # Extracting features from all the pre-processed images
    for filename in [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]:
        # Extract features from the image
        image_path = os.path.join(output_folder, filename)
        features = extracting_features(image_path)

        # Append the features to the list
        features_list.append(features)

    # Save the features to a CSV file
    # if no file exists, create one
    if not os.path.exists(output_file):
        save_features_to_csv(features_list, output_file)
    else:
        os.remove(output_file)
        save_features_to_csv(features_list, output_file)

if __name__ == '__main__':
    main()