import pandas as pd
import os
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, f1_score, roc_auc_score
import Functions2 as functions2
import pickle


# Load the data
data_path = "data\full_data.csv"
df = pd.read_csv(data_path)

# Preprocess the diagnostic column
df['diagnostic'] = df['diagnostic'].map({'BCC': 1, 'MEL': 1, 'SCC': 1, 'ACK': 0, 'NEV': 0, 'SEK': 0})

# Define the function to extract features
def extract_features(folder_path):
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    feature_5 = []
    feature_6 = []
    feature_7 = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            original = io.imread(image_path)

            # Ignore the alpha channel (e.g. transparency)
            if original.shape[-1] == 4:
                original = original[..., :3]

            feature_1.append(functions2.measure_pigment_network(original))
            feature_2.append(functions2.measure_blue_veil(original))
            feature_3.append(functions2.measure_vascular(original))
            feature_4.append(functions2.measure_globules(original))
            feature_5.append(functions2.measure_streaks(original))
            feature_6.append(functions2.measure_irregular_pigmentation(original))
            feature_7.append(functions2.measure_regression(original))

    return feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7

# Define the folder path for image processing
folder_path_in = "data\ColorMask\Training"

# Extract features from the images
feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7 = extract_features(folder_path_in)

# Create a DataFrame for the features
features_df = pd.DataFrame()
features_df["img_id"] = [filename for filename in os.listdir(folder_path_in) if filename.endswith(('.jpg', '.png'))]
features_df["1: pigment network"] = feature_1
features_df["2: Blue veil"] = feature_2
features_df["3: Vascular"] = feature_3
features_df["4: Globules"] = feature_4
features_df["5: Streaks"] = feature_5
features_df["6: Pigmentation"] = feature_6
features_df["7: Regression"] = feature_7

# Merge the features DataFrame with the diagnostic column from the original DataFrame
df_merged = pd.merge(df[['img_id', 'diagnostic']], features_df, on='img_id', how='inner')

# Split the data into training and testing sets
X = df_merged.drop(['img_id', 'diagnostic'], axis=1)
Y = df_merged['diagnostic']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

# Train the decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

# Make predictions on the test set
prediction = classifier.predict(X_test)

# Evaluate the decision tree classifier
cm = confusion_matrix(Y_test, prediction)
precision = precision_score(Y_test, prediction)
recall = recall_score(Y_test, prediction)
f1 = f1_score(Y_test, prediction)
auc_roc = roc_auc_score(Y_test, prediction)
classification_rep = classification_report(Y_test, prediction)

# Print the evaluation metrics
print("Confusion Matrix:")
print(cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)
print("Classification Report:")
print(classification_rep)

# Save the trained classifier
pickle.dump(classifier, open("Pickle/DecisionTree.pkl", "wb"))

# Load the trained model
loaded_model = pickle.load(open("CPickle/DecisionTree.pkl", "rb"))

# Assuming you have a new image to predict
image_path = "C:/Users/annam/Desktop/Vascular/Masked/image5.png"

# Extract features from the new image
new_image_features = extract_features(image_path)

# Make the prediction using the loaded model
prediction = loaded_model.predict(new_image_features)

# Convert the numerical prediction back to the corresponding label
predicted_label = 'malignant' if prediction[0] == 1 else 'benign'

# Print the predicted label
print("Predicted Label:", predicted_label)