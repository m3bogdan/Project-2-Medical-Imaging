import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

def train_classifier(data_path, features_df):
    """
    Trains a classifier using the provided data and extracted features.

    Args:
        data_path (str): Path to the data CSV file.
        features_df (pd.DataFrame): DataFrame containing the extracted features.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained logistic regression classifier.
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Preprocess the diagnostic column
    df['diagnostic'] = df['diagnostic'].map({'BCC': 1, 'MEL': 1, 'SCC': 1, 'ACK': 0, 'NEV': 0, 'SEK': 0})

    # Merge the features DataFrame with the diagnostic column from the original DataFrame
    df_merged = pd.merge(df[['img_id', 'diagnostic', 'patient_id']], features_df, on='img_id', how='inner')

    # Split the data into training and testing sets
    X = df_merged.drop(['img_id', 'diagnostic'], axis=1)
    Y = df_merged['diagnostic']

    # Different classifiers to test out
    classifiers = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier()]

    trained_classifiers = {}

    for classifier in classifiers:
        # Handle missing values in X
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Check for constant features or features with zero variance
        non_zero_var_indices = X_imputed.var(axis=0) != 0
        if not any(non_zero_var_indices):
            raise ValueError("All features have zero variance. Cannot perform PCA.")

        # Standardize the feature matrix
        X_std = StandardScaler().fit_transform(X_imputed[:, non_zero_var_indices])

        # Perform PCA and retain the first four principal components
        pca = PCA(0.99)
        X_pca = pca.fit_transform(X_std)

        # Initialize and train the model using the reduced feature space
        model = classifier.fit(X_pca, Y)

        # Save the trained classifier
        trained_classifiers[type(classifier).__name__] = model

    return trained_classifiers

# Path to the data and extracted features
data_path = r"data\full_data.csv"
features_csv_path = r"MAIN_FILES\MAIN_DATA\Input\features.csv"
model_path = r"MAIN_FILES\MAIN_DATA\Models"

# Load the extracted features from the CSV file
features_df = pd.read_csv(features_csv_path)

# Train the classifiers and get the trained models
trained_classifiers = train_classifier(data_path, features_df)

# Save the trained models
for classifier_name, classifier_model in trained_classifiers.items():
    model_path = f"{classifier_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(classifier_model, f)
