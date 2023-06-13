import pickle
import pandas as pd
from process_images import extract_features_folder
from sklearn.metrics import f1_score



def open_picke(picked_path):

    with open(picked_path, 'rb') as file:
        model = pickle.load(file)
    return model

#open_picke(r"Fixed\PROJECT\fyp2023\models\group02_classifier.sav")

def create_df(image_folder,csv_path):
    df = pd.read_csv(csv_path)
    features = extract_features_folder(image_folder)
    df_merged = pd.merge(df[['img_id', 'diagnostic']], features, on='img_id', how='inner')   
    return df_merged
#print(create_csv(r"C:\Users\45911\Desktop\DS\Semester2\First_year_project\Project2\Color_mask\Color_mask\Test",r"C:\Users\45911\Desktop\DS\Semester2\First_year_project\Project2\GitHub\Project-2-Medical-Imaging\Fixed\PROJECT\fyp2023\csv_datatypes\Bad.csv"))

def test(pickled_path,image_folder,csv_path):
    picked_file = open_picke(pickled_path)
    data = create_df(image_folder,csv_path)

    X = data.drop(['diagnostic', 'img_id'], axis=1)
    y = data['diagnostic']
    #y_predict =  picked_file.predict(X)
    print(data)
    #print(y_predict)
    #f1_sc = f1_score(y,y_predict)
    #print(f"the f1 score is. {f1_sc}")

    #return f1_sc
    return y
pickled_path = r"Fixed\PROJECT\fyp2023\models\group02_classifier.sav"
image_folder = r"C:\Users\45911\Desktop\DS\Semester2\First_year_project\Project2\Pictures\data\TestGood"
csv_path = r"Fixed\PROJECT\fyp2023\csv_datatypes\Bad.csv"
test(pickled_path,image_folder,csv_path)

