import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBClassifier
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler
from deepod.models import PReNet
from itertools import combinations
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score, precision_score, f1_score, confusion_matrix

features = ['Src IP', 'Dst IP','Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
    'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
    'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Byts',
    'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min']



def clean_df(df):
    # Remove the space before each feature names
    df = df[features].copy()
    df.columns = df.columns.str.strip()
    print('dataset shape', df.shape)
    

    # This set of feature should have >= 0 values
    num = df._get_numeric_data()
    num[num < 0] = 0

    zero_variance_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            zero_variance_cols.append(col)
    df.drop(zero_variance_cols, axis = 1, inplace = True)
    print('zero variance columns', zero_variance_cols, 'dropped')
    print('shape after removing zero variance columns:', df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    print(df.isna().any(axis = 1).sum(), 'rows dropped')
    df.dropna(inplace = True)
    print('shape after removing nan:', df.shape)

    # Drop duplicate rows
    df.drop_duplicates(inplace = True)
    print('shape after dropping duplicates:', df.shape)

    column_pairs = [(i, j) for i, j in combinations(df, 2) if df[i].equals(df[j])]
    ide_cols = []
    for column_pair in column_pairs:
        ide_cols.append(column_pair[1])
    df.drop(ide_cols, axis = 1, inplace = True)
    print('columns which have identical values', column_pairs, 'dropped')
    print('shape after removing identical value columns:', df.shape)
    return df

def sample_df(curr_df, anomaly_rate):
    num_benign = len(curr_df.loc[curr_df['Label'] == "BENIGN"])
    num_attack = len(curr_df) - num_benign
    ratio = num_attack / num_benign
    
    if ratio > anomaly_rate:
        sample = anomaly_rate * num_benign / num_attack
        sampled_df = pd.concat([curr_df[curr_df['Label'] == 'BENIGN'], 
                                curr_df[curr_df['Label'] != 'BENIGN'].sample(frac=sample, random_state=42)]) 
    else:
        
        sample = (1/anomaly_rate) * num_attack / num_benign
        sampled_df = pd.concat([curr_df[curr_df['Label'] != 'BENIGN'], 
                                curr_df[curr_df['Label'] == 'BENIGN'].sample(frac=sample, random_state=42)]) 
    
    new_ratio = sampled_df.loc[sampled_df["Label"] == "BENIGN"].shape[0] / sampled_df.loc[sampled_df["Label"] != "BENIGN"].shape[0]
    
    return sampled_df

def read_csv():
    full_df = pd.DataFrame()
    dataset_csv_path = './Dataset/SimulatedCVE/cicflowmeter_cve/'
    folder_names = ['dripper/', 'BENIGN/', 'bonesi/']
    
    for folder in folder_names:
        csv_file_names = os.listdir("Dataset/SimulatedCVE/cicflowmeter_cve/" + folder)
        complete_paths = []
        for csv_file_name in csv_file_names:
            print(csv_file_name)
            complete_paths.append(os.path.join(dataset_csv_path+folder, csv_file_name))
        df = pd.concat(map(pd.read_csv, complete_paths), 
                                ignore_index = True)
    
        df["Label"] = folder[:-1]
        full_df = pd.concat([full_df, df], axis=0, ignore_index=True)
    cleaned_df = clean_df(full_df)
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)  # Replace inf/-inf with NaN
    cleaned_df = cleaned_df.fillna(df.mean())
    cleaned_df.dropna(inplace=True)
    cleaned_df = cleaned_df.select_dtypes(include=[np.number])
    indices_to_keep = ~cleaned_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    cleaned_df = cleaned_df[indices_to_keep]
    return full_df
  
  
def validated_req_schema(flow_data):
  df_pruned = flow_data[features]
  return df_pruned  

def load_data(scaler):
  full_df = read_csv()
  columns = full_df.columns
  normalized_data = scaler.transform(full_df)
  normalized_df = pd.DataFrame(normalized_data, columns = columns)
  X_train, X_test, y_train, y_test = train_test_split(normalized_df.drop(["Label"], axis=1), normalized_df["Label"],
                                                      test_size=0.2, random_state=4022)
  return X_train, X_test, y_train, y_test
  

def get_optimal_threshold(precision, recall, thresholds):
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0  # Replace NaN values with 0    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold
  
def eval_accuracy(clf, X_test, y_test):
    anomaly_scores = clf.decision_function(X_test.to_numpy())
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    opt_threshold = get_optimal_threshold(precision, recall, thresholds)
    pred = np.where(anomaly_scores > opt_threshold, 1,0)
    f1 = f1_score(y_test, pred)
    conf_matrix = confusion_matrix(y_test, pred)
    print(f"F1 Score: {f1:.4f}")
    print(conf_matrix)
    return opt_threshold
    
    
class Global_Model():
  def __init__(self):
    self.scaler = joblib.load('cic_scaler.joblib')
    self.model, self.opt_threshold = self.load_model()
    
  def load_model(self): # Load the model through training since pytorch isn't supported
    model = PReNet
    clf = model(epochs=1, device='cuda')
    X_train, X_test, y_train, y_test = load_data(self.scaler)
    clf.fit(X_train.to_numpy(), y_train)
    
    opt_threshold = eval_accuracy(clf, X_test, y_test)
    return clf, opt_threshold
      
  def perform_inference_sup(self, X):
    X.to_csv("x.csv")

    X_scaled = self.scaler.transform(X)
    anomaly_scores = self.model.decision_function(X_scaled.to_numpy())
    output = np.where(anomaly_scores > self.opt_threshold, 1,0)
    print(np.unique(output)) 
    return output
