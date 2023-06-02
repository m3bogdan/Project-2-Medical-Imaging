import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
import warnings

def load_data(features_file_path, full_data_path):
    features_file = pd.read_csv(features_file_path)
    data_path = pd.read_csv(full_data_path)

    df = data_path[['img_id', 'diagnostic']]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)
        df['diagnostic'] = df['diagnostic'].replace({'BCC': 1, 'MEL': 1, 'SCC': 1, 'ACK': 0, 'NEV': 0, 'SEK': 0})

    merged_df = pd.merge(features_file, df, left_on='filename', right_on='img_id', how='left')
    merged_df = merged_df.drop('img_id', axis=1)

    return merged_df

def split_data(data):
    X = data[['pigment_network_coverage', 'blue_veil_pixels', 'vascular_pixels', 'globules_count', 'streaks_irregularity', 'irregular_pigmentation_coverage', 'regression_pixels']]
    y = data['diagnostic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test


def train_classifier_without_PCA(X_train, y_train):
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'DTC': DecisionTreeClassifier(),
    }
        
    trained_classifiers = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            trained_classifiers[name] = clf

    return trained_classifiers

def train_classifier_with_PCA(X_train, y_train):
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'DTC': DecisionTreeClassifier(),
    }
    
    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    trained_classifiers = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, clf in classifiers.items():
            clf.fit(X_train_pca, y_train)
            trained_classifiers[name] = clf

            clf_pca = clone(clf)
            clf_pca.fit(X_train_pca, y_train)
            trained_classifiers[f'PCA_{name}'] = clf_pca

    return trained_classifiers

def save_model(models, model_path):
    os.makedirs(model_path, exist_ok=True)
    for name, model in models.items():
        model_file = os.path.join(model_path, f'{name}.pickle')
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

def main():
    features_file_path = r'C:\Users\serru\Downloads\img\features.csv'
    full_data_path = r'C:\Users\serru\OneDrive\Documents\Project2\Project-2-Medical-Imaging\data\csv\full_data.csv'
    merged_data = load_data(features_file_path, full_data_path)
    #save merged data to csv
    merged_data.to_csv(r'C:\Users\serru\Downloads\img\merged_data.csv', index=False)
    X_train, X_test, y_train, y_test = split_data(merged_data)
    model_path = r"C:\Users\serru\Downloads\img\Models"
    models = train_classifier_without_PCA(X_train, y_train)
    save_model(models, model_path)
    print("Models saved to disk.")

if __name__ == '__main__':
    main()
