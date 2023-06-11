import pickle #for loading your trained classifier
from skimage import io
import os
from PIL import Image
from extract_features import extract_features #our feature extraction
import numpy as np
import matplotlib.pyplot as plt



# # The function that should classify new images. 
# # The image and mask are the same size, and are already loaded using plt.imread
# def classify(image, mask):

#     normal_image = image.Image.convert("RGBA")
#     segmentation_image = mask.Image.convert("RGBA")

#     #Resize the images to 256x256 pixels
#     # normal_image = image.resize((256, 256))
#     # segmentation_image = mask.resize((256, 256))

#     # Invert the segmentation image
#     inverted_segmentation = Image.eval(segmentation_image, lambda x: 255 - x)

#     # Create a binary mask from the inverted segmentation image
#     mask = inverted_segmentation.split()[0].point(lambda x: 255 if x == 0 else 0).convert("L")

#     # Apply the mask to the normal image
#     normal_image.putalpha(mask)

#     # Create a black background
#     background = Image.new("RGBA", normal_image.size, (0, 0, 0, 255))
#     # Composite the normal image with the black background
#     combined_image = Image.alpha_composite(background, normal_image)


#      #Extract features (the same ones that you used for training)
#     X = list(extract_features(combined_image).values())
#     X = np.array(X).reshape(1, -1)   
     
#      # Load the trained classifier
#     classifier = pickle.load(open('group02_classifier.sav', 'rb'))

#      # Use it on this example to predict the label AND posterior probability
#     pred_label = classifier.predict(X)
#     pred_prob = classifier.predict_proba(X)

#      # Get the probability for the predicted class
#     predicted_class_prob = pred_prob[0, pred_label[0]]

#     print('predicted label is', pred_label)
#     print('predicted probability for the predicted class is', predicted_class_prob)
     
#     return pred_label, predicted_class_prob

# normal_image_path ="C:/Users/annam/Desktop/Images/Resized/PAT_26_37_865.png"
# segmentation_image_path ="C:/Users/annam/Desktop/Binary masks/Resized/PAT_26_37_865.png"
# image = plt.imread(normal_image_path)
# mask = plt.imread(segmentation_image_path)


# classify(image, mask)


from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import pickle
import extract_features
import numpy as np
import os
from skimage import io


def classify(normal_image_path, segmentation_image_path):

    # Load and resize normal and segmentation images
    normal_image = io.imread(normal_image_path)
    if normal_image.shape[-1] == 4:  # Ignore the alpha channel (transparency)
        normal_image = normal_image[..., :3]
    normal_image = resize(normal_image, (256, 256), mode='reflect')

    segmentation_image = io.imread(segmentation_image_path)
    segmentation_image = resize(segmentation_image, (256, 256), mode='reflect')

    # Create binary mask from inverted segmentation image, thresholding at mid-intensity
    binary_mask = img_as_ubyte(rgb2gray(segmentation_image)) < 128

    # Apply the mask to the normal image
    combined_image = normal_image * binary_mask[..., None]

    # Extract features
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


normal_images_path = "C:/Users/annam/Desktop/Images/Resized"
segmentation_images_path = "C:/Users/annam/Desktop/Binary masks/Resized"

# Assume normal and segmentation images have the same filenames
for filename in os.listdir(normal_images_path):
    if filename.endswith(('.jpg', '.png')):
        normal_image_path = os.path.join(normal_images_path, filename)
        segmentation_image_path = os.path.join(segmentation_images_path, filename)

        if os.path.exists(segmentation_image_path):
            classify(normal_image_path, segmentation_image_path)


