import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import pickle

# Define relative path for the features
relative_path = "./Project-2_github_repo/Fixed/PROJECT/fyp2023/features/features.csv"

# Define classifiers
classifiers = [KNN(5), LR(max_iter=5000), DTC()]
classifier_names = ["KNN", "LR", "DTC"]
num_classifiers = len(classifiers)
num_folds = 5

# Read data
df_merged = pd.read_csv(relative_path)
x = np.array(df_merged.drop(['img_id', 'diagnostic', 'patient_id'], axis=1))
y = df_merged['diagnostic']
patient_id = df_merged['img_id']
group_kfold = GroupKFold(n_splits=num_folds)

# Define functions to avoid code repetition
def get_metrics(classifiers, x, y, patient_id, feature_selection=""):
    acc_val = np.empty([num_folds, num_classifiers])
    f1_val = np.empty([num_folds, num_classifiers])
    precision = np.empty([num_folds, num_classifiers])
    recall = np.empty([num_folds, num_classifiers])
    roc_auc = np.empty([num_folds, num_classifiers])
    for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
        x_train, x_val = x[train_index, :], x[val_index, :]
        y_train, y_val = y[train_index], y[val_index]
        for j, clf in enumerate(classifiers): 
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_val)
            acc_val[i, j] = accuracy_score(y_val, predictions)
            f1_val[i, j] = f1_score(y_val, predictions)
            precision[i, j] = precision_score(y_val, predictions, zero_division=0)
            recall[i, j] = recall_score(y_val, predictions)
            roc_auc[i, j] = roc_auc_score(y_val, predictions)
    print_metrics(acc_val, f1_val, precision, recall, roc_auc, feature_selection)
    return x_train, y_train, x_val, y_val  # return these for later use in ROC plot

def print_metrics(acc_val, f1_val, precision, recall, roc_auc, feature_selection):
    average_acc = np.mean(acc_val, axis=0) 
    average_f1 = np.mean(f1_val, axis=0)
    average_precision = np.mean(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    average_roc_auc = np.mean(roc_auc, axis=0)
    print(f"{feature_selection} feature selection:")
    for i, classifier_name in enumerate(classifier_names):
        print(f"############ Classifier {i+1} - {classifier_name}:")
        print(f'Average F1 score = {average_f1[i]:.3f}')
        print(f'Average Accuracy = {average_acc[i]:.3f}')
        print(f'Average Precision = {average_precision[i]:.3f}')
        print(f'Average Recall = {average_recall[i]:.3f}')
        print(f'Average ROC AUC = {average_roc_auc[i]:.3f}')

# Normal feature selection
print("Running without feature selection")
x_train, y_train, x_val, y_val = get_metrics(classifiers, x, y, patient_id)

# Feature selection with variance threshold
print("Running with variance threshold feature selection")
threshold_value = 0.1
selector = VarianceThreshold(threshold=threshold_value)
x_selected = selector.fit_transform(x)
get_metrics(classifiers, x_selected, y, patient_id, "With variance threshold")

# Feature selection with PCA
print("Running with PCA feature selection")
pca_transformer = PCA(n_components=5)
x_pca = pca_transformer.fit_transform(x)
get_metrics(classifiers, x_pca, y, patient_id, "With PCA")

# Define the path where to save the file
pickle_path = 'Project-2_github_repo/Fixed/PROJECT/fyp2023/model_group02'

# Chosen classifier
classifier = KNN(n_neighbors=5)
classifier = classifier.fit(x, y)
filename = 'group02_classifier.sav'

# Check if the path exists, if not, create it
if not os.path.exists(pickle_path):
    os.makedirs(pickle_path)

pickle.dump(classifier, open(os.path.join(pickle_path, filename), 'wb'))


# Plot ROC curve for best classifier
clf = classifier
clf.fit(x_train, y_train)
y_score = clf.predict_proba(x_val)
fpr, tpr, _ = roc_curve(y_val, y_score[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
