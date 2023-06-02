# Project 2: Medical Imaging
Members: Anna Gnat, Petya Petrova, Alexis Serruys, Bogdan Mihaila

## Dependencies 
pip install cv2 <br>
pip install scikit-image

## How to use it
### Step 1: Changes in 01_process_images.py
Line 118: You should provide a path for the folder where the normal, full images are. <br>
Line 119: You should provide a path for the folder where the binary mask are. <br>
Line 120: You should provde an output folder for where the color masked images will ne stored. <br>
Line 127: You should provide the path to the CSV file where the features will be stored. <br>

### Step 2: Changes in 02_train_classifier.py
Line 73: You should provide the path to the CSV previously created. (Line 127 in Step 1) <br>
Line 75: You should provide the path where the pickle file will be saved. <br>

### Step 3: Changes in 03_evaluate_classifier.py
Line 96: You should provide the path to the CSV previously created. (Line 127 in Step 1) <br>
Line 108: You should provide the folder path with the images you want to test the model on. <br>
Line 126: You should provide the path to the previously saved model. (Line 75 in Step 2) <br>

