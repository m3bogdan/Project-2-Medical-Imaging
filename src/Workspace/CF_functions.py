#Imports
import pandas as pd
from skimage import io
import os
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, f1_score, roc_auc_score
import pickle
import functions

########## CREATING A DATA FRAME FOR THE CLASSIFIERS ##########

def read_data(data_path):
    df = pd.read_csv(data_path)
    return df


def preprocess_labels(df):
    """Changes the different diagnosis of skin lesions to either 1 (cancerous)
    or 0 (non-cancerous)"""

    new_df = df[['img_id', 'diagnostic']]
    new_df.loc[new_df['diagnostic'].isin(['BCC', 'MEL', 'SCC']), 'diagnostic'] = 1
    new_df.loc[new_df['diagnostic'].isin(['ACK', 'NEV', 'SEK']), 'diagnostic'] = 0
    return new_df


def extract_and_create_features(folder_path_in):
    """ Creates a data frame with image ID and all the features extracted for each image """
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    feature_5 = []
    feature_6 = []
    feature_7 = []
    file_names = []

    # Iterate through all the jpg and png files in the selected folder and read the images
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        image_path = os.path.join(folder_path_in, filename)
        original = io.imread(image_path)
        # Ignore the alpha channel (make it only RGB)
        if original.shape[-1] == 4:
            original = original[..., :3]

        # Extract each feature from every image and append it to a list
        feature_1.append(functions.measure_pigment_network(original))
        feature_2.append(functions.measure_blue_veil(original))
        feature_3.append(functions.measure_vascular(original))
        feature_4.append(functions.measure_globules(original))
        feature_5.append(functions.measure_streaks(original))
        feature_6.append(functions.measure_irregular_pigmentation(original))
        feature_7.append(functions.measure_regression(original))

        # Append the file name to the list of file names
        file_names.append(filename)

    # Create the features DataFrame
    features_df = pd.DataFrame({'img_id': file_names})
    features_df["1: Pigment network"] = feature_1
    features_df["2: Blue veil"] = feature_2
    features_df["3: Vascular"] = feature_3
    features_df["4: Globules"] = feature_4
    features_df["5: Streaks"] = feature_5
    features_df["6: Pigmentation"] = feature_6
    features_df["7: Regression"] = feature_7

    return features_df


def merge_data_frames(df1, df2):
    df_merged = pd.merge(df1, df2, on='img_id', how='inner')
    return df_merged

####

########## TRAINING AND EVALUATING DIFFERENT CLASSIFIERS ##########

def split_data(df_merged, test_size=0.33):
    X = df_merged.copy()
    X.drop("img_id", axis=1, inplace=True)
    X.drop("diagnostic", axis=1, inplace=True)
    Y = df_merged["diagnostic"].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def train_decision_tree(X_train, Y_train):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)
    with open("C:/Users/annam/Desktop/Trained_classifiers/trained_TREE.pkl", 'wb') as file:
        pickle.dump(classifier, file)
    return classifier

def plot_tree(classifier, feature_names, class_names):
    fig = plt.figure(figsize=(8, 8))
    _ = tree.plot_tree(classifier, feature_names=feature_names, class_names=class_names)
    plt.show()

def train_logistic_regression(X_train, Y_train):
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    # Save the trained classifier to a file
    import pickle
    with open("C:/Users/annam/Desktop/Trained_classifiers/trained_LR.pkl", 'wb') as file:
        pickle.dump(classifier, file)
    return classifier


def train_knn(X_train, Y_train, n_neighbors=5):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_train, Y_train)
    with open("C:/Users/annam/Desktop/Trained_classifiers/trained_KNN.pkl", 'wb') as file:
        pickle.dump(classifier, file)
    return classifier


def evaluate_classifier(classifier, X_train, Y_train, X_test, Y_test):
    # Perform cross-validation
    scores = cross_val_score(classifier, X_train, Y_train, cv=5)
    mean_accuracy = scores.mean()

    # Calculate F1 score
    y_pred = classifier.predict(X_test)
    f1 = f1_score(Y_test, y_pred)

    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(Y_test, y_pred)

    # Print the evaluation metrics
    print("Cross-Validation Accuracy:", mean_accuracy)
    print("F1 Score:", f1)
    print("AUC-ROC Score:", auc_roc)


def predict_with_classifier(classifier_path, image_path):
    classifier = pickle.load(open(classifier_path, "rb"))
    image = io.imread(image_path)
    prediction = classifier.predict(image)
    return prediction