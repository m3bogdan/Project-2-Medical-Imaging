from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd
import os
import Functions2
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from skimage import io



# Load the data
data_path = "C:/Users/annam/Desktop/ITU/2nd_sem/02_First_Year_Project/2nd_project/Project-2-Medical-Imaging/data/full_data.csv"
df = pd.read_csv(data_path)


#Preprocess the diagnostic column
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

            feature_1.append(Functions2.measure_pigment_network(original))
            feature_2.append(Functions2.measure_blue_veil(original))
            feature_3.append(Functions2.measure_vascular(original))
            feature_4.append(Functions2.measure_globules(original))
            feature_5.append(Functions2.measure_streaks(original))
            feature_6.append(Functions2.measure_irregular_pigmentation(original))
            feature_7.append(Functions2.measure_regression(original))
    return feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7




# Define the folder path for image processing
folder_path_in = "C:/Users/annam/Desktop/ITU/2nd_sem/02_First_Year_Project/2nd_project/Project-2-Medical-Imaging/data/ColorMask/Training"

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
merged_df = pd.merge(patient_df, diagnostic_df, on='patient_id', how='inner')
print(df_merged)  # Print the merged DataFrame

# Check the number of samples in the merged DataFrame
print("Number of samples:", len(df_merged))

# Split the data into training and testing sets
X = df_merged.drop(['img_id', 'diagnostic'], axis=1)
Y = df_merged['diagnostic']

# Print the shape of X and Y
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Split the data into training and testing sets
X = df_merged.drop(['img_id', 'diagnostic'], axis=1)
Y = df_merged['diagnostic']


classifiers = [LR(), KNN(), DTC()]  # Replace with your trained classifiers

results = []

# Perform cross-validation for each classifier
for classifier in classifiers:
    y_pred = cross_val_predict(classifier, X, Y, cv=5)  # Change cv value as per your requirement


    # Calculate evaluation metrics
    f1 = f1_score(Y, y_pred)
    precision = precision_score(Y, y_pred)
    recall = recall_score(Y, y_pred)
    tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Create a dictionary of the results for the current classifier
    result = {
        'Classifier': type(classifier).__name__,
        'F1 Score': f1,
        'Precision': precision,
        'Recall/Sensitivity': recall,
        'Specificity': specificity,
        'Confusion Matrix': confusion_matrix(Y, y_pred)
    }
    results.append(result)

# Convert the results list into a pandas DataFrame
results_df = pd.DataFrame(results)

# Print the results table
print(results_df)


