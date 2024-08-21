import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler
from deepod.models import PReNet
from itertools import combinations
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score, precision_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from alipy import ToolBox
from sklearn.tree import DecisionTreeClassifier


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
            complete_paths.append(os.path.join(dataset_csv_path+folder, csv_file_name))
        df = pd.concat(map(pd.read_csv, complete_paths), 
                                ignore_index = True)
        df = df[features].copy()
        print(folder[:-1])
        df["Label"] = folder[:-1]
        df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
        full_df = pd.concat([full_df, df], axis=0, ignore_index=True)
    label = full_df["Label"]    
    
    cleaned_df = clean_df(full_df)
    # Drop String Columns
    cleaned_df = cleaned_df[features]
    # cleaned_df = cleaned_df.drop(columns=['Src IP', 'Dst IP', "Label"])
    # Remove Inf and Nan
    # cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)  # Replace inf/-inf with NaN
    # cleaned_df = cleaned_df.fillna(df.mean())
    # cleaned_df.dropna(inplace=True)
    # cleaned_df = cleaned_df.select_dtypes(include=[np.number])
    # indices_to_keep = ~cleaned_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    # cleaned_df = cleaned_df[indices_to_keep]
    # Reinsert the label
    cleaned_df["Label"] = label
    cleaned_df.to_csv("x.csv")
    return cleaned_df
  
  
def validated_req_schema(flow_data):
  df_pruned = flow_data[features]
  return df_pruned  

def load_data(scaler):
  full_df = read_csv()
  # validated_df = validated_req_schema(full_df)
  label = full_df["Label"].values
  full_df.drop(columns=["Src IP", "Dst IP", "Label"], axis=1, inplace=True)
  columns = full_df.columns
  normalized_data = scaler.transform(full_df)
  normalized_df = pd.DataFrame(normalized_data, columns = columns)
  X_train, X_test, y_train, y_test = train_test_split(normalized_df, label,
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
  # TODO: New Data Handling - Append the data to the gm_training.csv
    
  def load_model(self): # Load the model through training since pytorch isn't supported
    model = PReNet
    clf = model(epochs=1, device='cuda')
    X_train, X_test, y_train, y_test = load_data(self.scaler)
    clf.fit(X_train.to_numpy()[:], y_train[:])
    
    opt_threshold = eval_accuracy(clf, X_test, y_test)
    return clf, opt_threshold
      
  def perform_inference(self, X):
    X_scaled = self.scaler.transform(X) # ! Add Scaler
    anomaly_scores = self.model.decision_function(X_scaled)
    output = np.where(anomaly_scores > self.opt_threshold, 1,0)
    print(np.unique(output)) 
    return output

class Local_Model():
    def __init__(self, gm):
        # self.model = XGBClassifier(objective='binary:logistic')
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=5,  
                                            min_samples_leaf=10, 
                                            # ccp_alpha=0.01, #Pruning coef
                                            random_state=4022)
        self.train_folder = "./Dataset/SimulatedCVE/cicflowmeter_cve/training_data/lm/"
        self.model_path = "./cic_xgb.joblib"
        self.state = 0 #0: OFF, 1: ON, 2: HYBRID
        self.scaler = joblib.load('./cic_scaler.joblib')
        self.global_model = gm #Replace With HTTP API
        self.load_model()
    def load_model(self):
        # self.model = joblib.load(self.model_path)
        known_df = self.load_known_df()
        y_known = known_df["Label"]
        X_known = known_df.drop(columns=["Label"], inplace=False)
        X_known_scaled = self.scaler.transform(X_known)
        self.model.fit(X_known_scaled, y_known)
        
    def load_known_df(self):
        known_df = pd.DataFrame()
        training_data_list = os.listdir(self.train_folder)
        training_data_list.sort(reverse=True)
        known_df =  pd.read_csv(os.path.join(self.train_folder,training_data_list[0]))
        print(training_data_list[0])
        # if len(training_data_list) > 0:
        #     print(training_data_list)
        #     for training_data in training_data_list:
        #         curr_df = pd.read_csv(os.path.join(self.train_folder,training_data))
        #         known_df = pd.concat([known_df, curr_df], axis=0, ignore_index=True)      
        # known_df.drop(columns=["Src IP", "Dst IP"], inplace=True)
        return known_df  
                
    def retrain_model(self, X_new): #Select Most Important Data and Upload Newly Recorded Data
        # TODO: Use AL to Select Prerecorded Data
        known_df = self.load_known_df()
        filtered_new_data, informative_score_list = self.select_data(known_df, X_new)
        # labeled_new_data = self.upload_gm(filtered_new_data)
        self.append_training_data(filtered_new_data)
        # TODO Update the model        
        
        return informative_score_list
        
    def upload_gm(self, X_query): 
        # X_query_scaled = self.scaler.transform(X_query)
        pseudo_label = self.global_model.perform_inference(X_query)
        X_query_df = pd.DataFrame(X_query, columns=features[2:])
        X_query_df["Label"] = pseudo_label
        return pseudo_label
    def select_data(self, known_df, X_new):
        round = 10
        threshold = 0.2  
        informative_score_list = []
        y_new = np.ones(X_new.shape[0]) * 2   
        model = self.model
        
        X_known = known_df.drop(columns=["Label"], inplace = False).copy()
        y_known = known_df["Label"]
        # X_known.drop(columns=["Src IP", "Dst IP"], inplace=True)
        X = np.concatenate([X_known, X_new])
        y = np.concatenate([y_known, y_new])
        label_ind = np.arange(len(X_known))
        print("Size of Label Index", len(X_known))
        unlab_ind = np.arange(len(X_known), len(X))

        divided_arrays = np.array_split(unlab_ind, round)
        
        alibox_new = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None,) 
        alibox_new.split_AL(test_ratio=0.1, initial_label_rate=0.1, split_count=10)     
        strategy_name = "QueryInstanceUncertainty"
        strategy = alibox_new.get_query_strategy(strategy_name=strategy_name) #TODO Replace Alibox with a single function
        for i in range(round):
            batch_size = 5000 
            print(f"Round {i}")
            # Use AL to Select Data
            select_ind, informative_score = strategy.select(label_index=label_ind, unlabel_index=divided_arrays[i], custom = True, model=model, batch_size=batch_size)
            select_ind = np.where(np.array(informative_score) > threshold)[0]
            informative_score.sort(reverse=True)
                              
            batch_size = min(batch_size, np.shape(select_ind)[0] ) #Limit up to 30 000 per query
            print(batch_size)
            # Upload Data to GM
            idx_to_query = select_ind[:batch_size]
            if len(idx_to_query) > 0:
                pseudo_labels = self.upload_gm(X[idx_to_query])
                y[idx_to_query] = pseudo_labels
                
                label_ind = np.concatenate([label_ind, select_ind[:batch_size]])# label_ind.update(select_ind)
                mask = np.where(np.isin(unlab_ind, select_ind[:batch_size], invert=True)) # unlab_ind.difference_update(select_ind)
                unlab_ind = unlab_ind[mask]
                
                print(f"Added {batch_size} Shape of Label_ind: {np.shape(label_ind)}")  
                #Update The Model
                X_scaled = self.scaler.transform(X[label_ind,:]) # ! Add Scaler
                
                model = XGBClassifier(objective="binary:logistic")
                model.fit(X=X_scaled, y=y[label_ind]) 
                # pred = model.predict(X_test)
                # query_accuracy = metric(pred, y_test)      
            else:
                print("No Data Added")
            print(informative_score[:100])
            informative_score_list.append(informative_score)

            
        merged_train_df = pd.DataFrame(X[label_ind,:], columns = features[2:])
        merged_train_df["Label"] = y[label_ind]
        # new_train_df = merged_train_df.iloc[len(X_known):]
        return merged_train_df, informative_score_list
        
    def perform_inferece(self):
            X_scaled = self.scaler.transform(X) # ! Add Scaler
            X_scaled = X
            output = self.model.predict(X_scaled) 
            print(np.unique(output)) 
            return output
    def append_training_data(self,new_train_df):
        #Write a new CSV FIle
        folder = self.train_folder
        filenumber = 0
        filepath = os.path.join(folder,f"lm_train_data_{filenumber}.csv")
        while os.path.exists(filepath):
            filenumber+=1
            filepath = os.path.join(folder, f"lm_train_data_{filenumber}.csv")
        new_train_df.to_csv(filepath, index=False)
        print(f"Added {filepath} as New LM Training Data")
        