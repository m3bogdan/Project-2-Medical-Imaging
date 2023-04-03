# Project 2: Medical Imaging
## Tasks

### Task 1A: segment images 
Create segmentations for some images. You can do this with image processing methods, or yourself with LabelStudio

### Task 1B: measure the features yourself 
Search for related work about the Asymmetry and Color features and how they are measured by dermatologists. Create an “annotation guide” for you and your group members, where you discuss at least 5 images together, and decide how to rate their Asymmetry and Color. Then split the images, such that each image is annotated by at least two people in your group. Save your annotations in a CSV file, such that there
are as many columns as there are different annotators (+ one column for the image name), i.e. do not put annotations of diffferent people into the same column. Make sure your CSV file follows the guidelines outlined in [?].

### Task 2: measure the features with an algorithm
Create implementations for the Asymmetry and Color features using related work in image analysis. There will be multiple (similar) ways to measure each feature, if this is the case you can motivate which method you choose. You may use code available online but you need to be able to explain and modify different steps of the code. To test your implementations, you might want to create “toy” images where you already know the results, for example a circle should be less asymmetric than an ellipse, etc. Once you are satisfied with your implementations, run them on the real images and save the features in a CSV file. Compare the features to your manual measurements by calculating agreement and/or visualizing the measurements. Do you agree with your algorithm? Do you see any other patterns?

### Task 3: predict the diagnosis
For this task, you can use more images from the same dataset, or use other public data sources that you find. Use a cross-validation setup to train different classifiers we studied in class (no neural networks), and evaluate their performance with appropriate metrics. You may also use other ways of evaluating classifiers, for example inspecting images that are classified incorrectly. After this, select your best set of features + classifier. Train this classifier on the entire dataset (without cross-validation) and save the trained classifier. Then create a function that can classify an external image/mask. This function should measure features you used, apply any transformations etc,
and finally apply your trained classifier. The classifier should output a probability of the image being suspicious, between 0 (healthy) and 1 (not healthy). This will be evaluated on a different set of data, which is not given to you. The external data will have masks
available.

### Task 4: open question
Use the data and your findings so far to formulate, motivate, answer, and discuss another research question of your choice. For example, you can study additional datasets, differences between groups of patients, additional types of features, etc.
