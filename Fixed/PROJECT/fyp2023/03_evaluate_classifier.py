import pickle
import numpy as np
from extract_features import extract_features
from PIL import Image
import matplotlib.pyplot as plt
import os

def classify(normal_image, segmentation_image):
    # Make sure the images are the same size
    assert normal_image.shape == segmentation_image.shape, "Images are not the same size"

    # Convert numpy array image to PIL image
    normal_image = Image.fromarray((normal_image * 255).astype(np.uint8)).convert("RGBA")
    segmentation_image = Image.fromarray((segmentation_image * 255).astype(np.uint8)).convert("RGBA")

    # Invert the segmentation image
    inverted_segmentation = Image.eval(segmentation_image, lambda x: 255 - x)

    # Create a binary mask from the inverted segmentation image
    mask = inverted_segmentation.split()[0].point(lambda x: 255 if x == 0 else 0).convert("L")

    # Apply the mask to the normal image
    normal_image.putalpha(mask)

    # Create a black background
    background = Image.new("RGBA", normal_image.size, (0, 0, 0, 255))

    # Composite the normal image with the black background
    combined_image = Image.alpha_composite(background, normal_image)

    # Convert the PIL Image object back to numpy array and convert to RGB if needed
    combined_image = np.array(combined_image.convert("RGB"))

    # Extract features (the same ones that you used for training)
    X = list(extract_features(combined_image).values())
    X = np.array(X).reshape(1, -1)   

    # Define the path where the classifier is located
    pickle_path = "Project-2_github_repo/Fixed/PROJECT/fyp2023/models"

    # Load the trained classifier
    classifier = pickle.load(open(os.path.join(pickle_path, "group02_classifier.sav"), 'rb'))

    # Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(X)
    pred_prob = classifier.predict_proba(X)

    # Get the probability for the predicted class
    predicted_class_prob = pred_prob[0, pred_label[0]]

    # Define label names
    label_names = ["Healthy", "Cancerous"]

    # Print the results
    print(f"The image is predicted to be {label_names[pred_label[0]]}")
    print(f"The probability from 0-1 (0 being healthy and 1 cancerous) is: {predicted_class_prob}")


# Define the path where the images are located
image_path = "Project-2_github_repo/Fixed/PROJECT/fyp2023/Predictions"

# Your image paths
normal_image_path = os.path.join(image_path, "OriginalImages", "PAT_26_37_865.png")
segmentation_image_path = os.path.join(image_path, "SegmentedImages", "PAT_26_37_865.png")


# Load the images
normal_image = plt.imread(normal_image_path)
segmentation_image = plt.imread(segmentation_image_path)

# Use the function
classify(normal_image, segmentation_image)
