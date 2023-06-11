import pickle
import numpy as np
from extract_features import extract_features
from PIL import Image
import matplotlib.pyplot as plt

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

    # Load the trained classifier
    classifier = pickle.load(open('group02_classifier.sav', 'rb'))

    # Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(X)
    pred_prob = classifier.predict_proba(X)

    # Get the probability for the predicted class
    predicted_class_prob = pred_prob[0, pred_label[0]]

    print('predicted label is', pred_label)
    print('predicted probability for the predicted class is', predicted_class_prob)

    return pred_label, predicted_class_prob

# Your image paths
normal_image_path = r"C:\Users\serru\Downloads\img\Test_resized\PAT_26_37_865.png"
segmentation_image_path = r"C:\Users\serru\Downloads\img\Binary_mask\Test\PAT_26_37_865.png"

# Load the images
normal_image = plt.imread(normal_image_path)
segmentation_image = plt.imread(segmentation_image_path)

# Use the function
classify(normal_image, segmentation_image)
