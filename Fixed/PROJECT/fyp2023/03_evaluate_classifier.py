import pickle
import numpy as np
from extract_features import extract_features
from PIL import Image
from extract_features import extract_features #our feature extraction
import numpy as np



# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(normal_image_path, segmentation_image_path):
        
    # Open the normal image and the segmentation image
    normal_image = Image.open(normal_image_path).convert("RGBA")
    segmentation_image = Image.open(segmentation_image_path).convert("RGBA")

    # Resize the images to 256x256 pixels
    normal_image = normal_image.resize((256, 256))
    segmentation_image = segmentation_image.resize((256, 256))

    # Invert the segmentation image
    inverted_segmentation = Image.eval(segmentation_image, lambda x: 255 - x)

    # Create a binary mask from the inverted segmentation image
    mask = inverted_segmentation.split()[0].point(lambda x: 255 if x == 0 else 0).convert("L")

    # Apply the mask to the normal image
    combined_image = normal_image * binary_mask[..., None]

    # Create a black background
    background = Image.new("RGBA", normal_image.size, (0, 0, 0, 255))
    # Composite the normal image with the black background
    combined_image = Image.alpha_composite(background, normal_image)


     #Extract features (the same ones that you used for training)
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

normal_image_path =""
segmentation_image_path =""
test_path = "C:/Users/annam/Desktop/Binary masks/Resized"
for filename in os.listdir(test_path):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(test_path, filename)
        image = io.imread(image_path)

        # Ignore the alpha channel (e.g. transparency)
        if image.shape[-1] == 4:
            image = image[..., :3]
