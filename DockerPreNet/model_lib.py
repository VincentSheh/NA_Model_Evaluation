import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from deepod.models import PReNet
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from alipy import ToolBox
from collections import Counter
from io import StringIO
from flask import Flask, request, jsonify
import json
import gc

features = ['Src IP', 'Dst IP','Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
       'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
       'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
       'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
       'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
       'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
       'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
       'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
       'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
       'Pkt Len Var', 'FIN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
       'URG Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
       'Bwd Seg Size Avg', 'Subflow Fwd Byts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


def clean_df(df):
    # Remove the space before each feature names
    df.columns = df.columns.str.strip()
    print('dataset shape', df.shape)
    

    # This set of feature should have >= 0 values
    num = df._get_numeric_data()
    num[num < 0] = 0

    df = df.replace([np.inf, -np.inf], np.nan, inplace=False)  # Replace inf/-inf with NaN
    print(df.isna().any(axis = 1).sum(), 'rows dropped')
    df.dropna(inplace=True)
    # print('shape after removing nan:', df.shape)
    # Drop duplicate rows
    df.drop_duplicates(inplace = True)
    # print('shape after dropping duplicates:', df.shape) 
    
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf/-inf with NaN
    df = df.fillna(df.mean())       
    
    df.dropna(inplace=True)
    df = df.select_dtypes(include=[np.number])
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]

    for i in df.columns:
        df = df[df[i] != "Infinity"]
        df = df[df[i] != np.nan]
        df = df[df[i] != np.inf]
        df = df[df[i] != -np.inf]
        df = df[df[i] != ",,"]
        df = df[df[i] != ", ,"]
        
    # print(np.any(np.isnan(df)))
    # print(np.any(np.isfinite(df)))    

    return df

def sample_df(curr_df, anomaly_rate):
    num_benign = len(curr_df.loc[curr_df['Label'] == 0])
    num_attack = len(curr_df) - num_benign
    ratio = num_attack / num_benign
    
    if ratio > anomaly_rate:
        sample = anomaly_rate * num_benign / num_attack
        sampled_df = pd.concat([curr_df[curr_df['Label'] == 0], 
                                curr_df[curr_df['Label'] != 0].sample(frac=sample, random_state=42)]) 
    else:
        
        sample = (1/anomaly_rate) * num_attack / num_benign
        sampled_df = pd.concat([curr_df[curr_df['Label'] != 0], 
                                curr_df[curr_df['Label'] == 0].sample(frac=sample, random_state=42)]) 
    
    new_ratio = sampled_df.loc[sampled_df["Label"] == "0"].shape[0] / sampled_df.loc[sampled_df["Label"] != 0].shape[0]
    
    return sampled_df


def read_csv(folder_names = ['dripper/', 'BENIGN/', 'bonesi/']):
    full_df = pd.DataFrame()
    dataset_csv_path = './Dataset/SimulatedCVE/cicflowmeter_cve/'
    for folder in folder_names:

        csv_file_names = os.listdir("Dataset/SimulatedCVE/cicflowmeter_cve/" + folder)
        complete_paths = []
        for csv_file_name in csv_file_names:
            complete_paths.append(os.path.join(dataset_csv_path+folder, csv_file_name))
        print(complete_paths)
        df = pd.concat(map(pd.read_csv, complete_paths), ignore_index = True)
        if folder == 'training_data/gm/': #Avoid Dst IP and Src IP when loading from training folder
            df = df[features[2:]].copy()
        else:
            df = df[features[2:]].copy()
        print(folder[:-1])
        df["Label"] = folder[:-1]
        df["Label"] = df["Label"].apply(lambda x: 0 if (x == "BENIGN" or x == 0) else 1)
        full_df = pd.concat([full_df, df], axis=0, ignore_index=True)
    label = full_df["Label"]    
    cleaned_df = full_df.drop(columns=["Label"], inplace=False)
    cleaned_df = clean_df(cleaned_df)
    # Drop String Columns
    cleaned_df["Label"] = label
    return cleaned_df
  
def validated_req_schema(request, is_json = False, is_labeled = False): #is_labeled means for training
    # Check if a file is part of the POST request
    if is_labeled:
        features.append("Label")
    if is_json:
        data = request.get_json()
        if isinstance(data, list):
            flow_df = pd.DataFrame(data)
        else:
            # If needed, use pd.read_json with properly formatted JSON string
            flow_df = pd.read_json(data, orient="records")
        if data is None:
            return jsonify({"error": "No data received"}), 400
        df_pruned = flow_df[features]
        return df_pruned
    
    if 'file' not in request.files:
        print("No file part")
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return "No selected file", 400
    if file:
        # Convert the file stream directly to a DataFrame
        string_data = StringIO(file.read().decode('utf-8'))   
        flow_df = pd.read_csv(string_data)   
          
    df_pruned = flow_df[features]
    return df_pruned  

def get_optimal_threshold(precision, recall, thresholds):
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0  # Replace NaN values with 0    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold
  
def eval_accuracy(clf, X_test, y_test):
    # anomaly_scores = []   
    # for i in range(0, X_test.shape[0], batch_size) :
    #     batch_X = X_test[i:i+batch_size]
    #     batch_scores = clf.decision_function(batch_X.to_numpy())
    #     anomaly_scores.append(batch_scores)
    # anomaly_scores = np.concatenate(anomaly_scores)
    
    anomaly_scores = clf.decision_function(X_test.to_numpy())
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    opt_threshold = get_optimal_threshold(precision, recall, thresholds)
    pred = np.where(anomaly_scores > opt_threshold, 1,0)
    f1 = f1_score(y_test, pred)
    conf_matrix = confusion_matrix(y_test, pred)
    print(f"F1 Score: {f1:.4f}, Accuracy: {accuracy_score(pred, y_test):.4f}")
    print(conf_matrix)
    return opt_threshold

def get_avail_filename(folder,filename):
    filenumber = 0
    filepath = os.path.join(folder,f"{filename}_{filenumber}.csv")
    while os.path.exists(filepath):
        
        filenumber+=1
        filepath = os.path.join(folder, f"{filename}_{filenumber}.csv")  
    print(filepath)  
    return filepath

def load_data(train_folder,scaler=None):
    full_df = read_csv(train_folder)
    # validated_df = validated_req_schema(full_df)
    label = full_df["Label"].values
    full_df.drop(columns=["Src IP", "Dst IP", "Label"], axis=1, inplace=True)
    columns = full_df.columns
    if scaler != None:
        normalized_data = scaler.transform(full_df)
        normalized_df = pd.DataFrame(normalized_data, columns = columns)
    else:
        normalized_df = full_df.copy()
    X_train, X_test, y_train, y_test = train_test_split(normalized_df, label, shuffle=True, stratify=label,
                                                        test_size=0.2, random_state=4022)
    return X_train, X_test, y_train, y_test    

    
class Global_Model():
  def __init__(self, train_folder = ['dripper/', 'BENIGN/', 'bonesi/'], new_data_folder = ["training_data/gm/"]):
    self.scaler = joblib.load('cic_scaler.joblib')
    self.train_folder = train_folder
    self.new_train_folder = new_data_folder
    self.model, self.opt_threshold = self.load_model()
    
    
  def load_data(self,scaler=None):
    full_df = read_csv(self.train_folder)

    # validated_df = validated_req_schema(full_df)
    sampled_df = sample_df(full_df,0.05) # !Shouldn't Sample when loading from ./training_data/
    print("Before Merging", sampled_df["Label"].value_counts())
    if os.listdir("./Dataset/SimulatedCVE/cicflowmeter_cve/" + self.new_train_folder[0]):
        new_train_df = read_csv(self.new_train_folder)    
        sampled_df = pd.concat([sampled_df, new_train_df], ignore_index=True)
    label = sampled_df["Label"].values
    
    print("After Merging", sampled_df["Label"].value_counts())
    sampled_df.drop(columns=["Label"], axis=1, inplace=True)
    columns = sampled_df.columns
    if scaler != None:
        normalized_data = scaler.transform(sampled_df)
        normalized_df = pd.DataFrame(normalized_data, columns = columns)
    else:
        normalized_df = sampled_df.copy()
    # For Training Include the Unsupervised Labeled Data
    X_train, X_test, y_train, y_test = train_test_split(normalized_df, label, shuffle=True, stratify=label,
                                                        test_size=0.2, random_state=4022)
    
    return X_train, X_test, y_train, y_test    
    
  def load_model(self, eval_flag=True): # Load the model through training since pytorch isn't supported
    model = PReNet
    clf = model(epochs=1, device='cpu', batch_size=32)
    # if eval_flag:
    X_train, X_test, y_train, y_test = self.load_data(self.scaler)
    clf.fit(X_train.to_numpy()[:20000], y_train[:20000])
    
    opt_threshold = eval_accuracy(clf, X_test, y_test) #! Should run this line, when initally load model
    gc.collect()
    return clf, opt_threshold
    # self.model = clf
            
      
  def perform_inference(self, X):
    batch_size = 64
    X_scaled = self.scaler.transform(X)  # Scale the input data
    anomaly_scores = []

    # Process in batches
    # for i in range(0, X.shape[0], batch_size):
    #     batch_X = X_scaled[i:i + batch_size]  # Select the batch
    #     batch_scores = self.model.decision_function(batch_X)  # Get the scores for the batch
    #     anomaly_scores.append(batch_scores)  # Append the batch scores
    # anomaly_scores = np.concatenate(anomaly_scores)
    anomaly_scores = self.model.decision_function(X_scaled)    
    output = np.where(anomaly_scores > self.opt_threshold, 1,0)
    unique, counts = np.unique(output, return_counts=True)
    print(f"Malicious Request: {counts[0]} , Benign Request:{counts[1]}")
    gc.collect()
    return output

  def update_data(self,X, folder_names = None):
    if folder_names == None: 
        folder_names = self.new_train_folder[0] #"gm_train_data"
    #Write a new CSV FIle
    file = "gm_train_data"
    folder_names = os.path.join("Dataset/SimulatedCVE/cicflowmeter_cve/",self.new_train_folder[0]) #"gm_train_data"
    filename = get_avail_filename(folder_names, file)
    filepath = filename
    if not X.empty:
        X.to_csv(filepath, index=False)
    print(f"Added {filepath} as New GM Training Data")
    
  def gm_select_data(self, data):
    scaled_data = data[features[2:]]
    scaled_data = self.scaler.transform(scaled_data)
    print(scaled_data)
    scores = self.model.decision_function(scaled_data) #Perform Active Learning Selection
    selected_idx = np.where(np.logical_and(scores > self.opt_threshold -2, scores < self.opt_threshold +1))
    print(f"\033[32mSelected {len(selected_idx)}\033[0m")
    return data.iloc[selected_idx]
      
  def retrain_gm(self, X = pd.DataFrame()):
    if not X.empty:
        filtered_data = self.gm_select_data(X)
        self.update_data(filtered_data)
        self.load_model(eval_flag=False) # ? Reload The Model
    else:
        print("No Training Data Added to GM")
  def compress_training_data(self):
    pass

class Local_Model():
    def __init__(self):
        self.model = XGBClassifier(objective='binary:logistic')
        self.train_folder = "./Dataset/SimulatedCVE/cicflowmeter_cve/training_data/lm/"
        self.model_path = "./cic_xgb.joblib"
        self.state = 0 #0: OFF, 1: ON, 2: HYBRID
        self.scaler = joblib.load('./cic_scaler.joblib')
        # self.load_model()
        self.model = joblib.load(self.model_path)
    def load_model(self):
        # self.model = joblib.load(self.model_path)
        known_df = self.load_known_df()
        y_known = known_df["Label"]
        X_known = known_df.drop(columns=["Label"], inplace=False)
        X_known = X_known[features[2:]]
        X_known_scaled = self.scaler.transform(X_known)
        self.model.fit(X_known_scaled, y_known)
        
    def load_known_df(self):
        known_df = pd.DataFrame()
        training_data_list = os.listdir(self.train_folder)
        training_data_list.sort(reverse=True)
        # known_df =  pd.read_csv(os.path.join(self.train_folder,training_data_list[0]))
        # print(training_data_list[0])
        if len(training_data_list) > 0:
            print(training_data_list)
            for training_data in training_data_list:
                curr_df = pd.read_csv(os.path.join(self.train_folder,training_data))
                known_df = pd.concat([known_df, curr_df], axis=0, ignore_index=True)      
        # known_df.drop(columns=["Src IP", "Dst IP"], inplace=True)
        return known_df  
                
    def retrain_model(self, X_new, y_new=None, threshold = 0.2, update_gm = False): #Select Most Important Data and Upload Newly Recorded Data
        # TODO: Use AL to Select Prerecorded Data
        known_df = self.load_known_df()
        filtered_new_data, updated_model = self.select_data(known_df, X_new, threshold, y_new)
        # labeled_new_data = self.upload_gm(filtered_new_data)
        if update_gm:
            filtered_new_data["Label"] = 0
            # self.global_model.update_data(filtered_new_data)
        # TODO: After updating return the labels or recall the function
        else:
            self.append_training_data(filtered_new_data)
            # Update the model        
            self.model = updated_model # ! If update_gm, should replace the previous record for AL to work
            print("\033[36mModel Successfully Updated\033[0m")
        # return informative_score_list
        
    def upload_gm(self, X_query): 
        # X_query_scaled = self.scaler.transform(X_query)
        pseudo_label = self.global_model.perform_inference(X_query)
        X_query_df = pd.DataFrame(X_query, columns=features[2:])
        X_query_df["Label"] = pseudo_label
        return pseudo_label
    def select_data(self, known_df, X_new, threshold, y_new):
        round = 10
        informative_score_list = []
        y_new = np.ones(X_new.shape[0]) * 2 if y_new is None else y_new
        model = self.model
        X_known = known_df.drop(columns=["Label"], inplace = False).copy()
        y_known = known_df["Label"]
        # Append 20% of Data Labeled MALICIOUS from the new training data #!There might be a bug here
        selected_malicious_idx = y_new[y_new==1].sample(frac=0.2, random_state=4022).index
        X_known = pd.concat([X_known, X_new.iloc[selected_malicious_idx]], ignore_index=True)
        y_known = np.concatenate([y_known, y_new[selected_malicious_idx]]).astype(int)
        print(f"Added {len(selected_malicious_idx)} to training data")
        X_new.drop(selected_malicious_idx, inplace=True)
        y_new.drop(selected_malicious_idx, inplace=True)
        # X_known.drop(columns=["Src IP", "Dst IP"], inplace=True)
        X = pd.concat([X_known, X_new], ignore_index=True)
        y = np.concatenate([y_known, y_new]).astype(int)
        label_ind = np.arange(len(X_known))
        print("Size of Label Index", len(X_known))
        unlab_ind = np.arange(len(X_known), len(X))

        divided_arrays = np.array_split(unlab_ind, round)
        
        alibox_new = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None,) 
        alibox_new.split_AL(test_ratio=0.1, initial_label_rate=0.1, split_count=10)     
        strategy_name = "QueryInstanceUncertainty"
        strategy = alibox_new.get_query_strategy(strategy_name=strategy_name) #TODO Replace Alibox with a single function
        for i in range(round):
            batch_size = 10000 
            print(f"Round {i}")
            # Use AL to Select Data
            select_ind = strategy.select(label_index=label_ind, unlabel_index=divided_arrays[i], 
                                                            threshold=threshold, custom = True, model=model, batch_size=batch_size)
            batch_size = min(batch_size, np.shape(select_ind)[0] ) #Limit up to 30 000 per query
            # Upload Data to GM
            idx_to_query = select_ind[:batch_size]
            if len(idx_to_query) > 0:
                if 2.0 in y_new.values: 
                    pseudo_labels = self.upload_gm(X.iloc[idx_to_query]) 
                    y[idx_to_query] = pseudo_labels 
                else: # ? Set GM to be 100% Accurate
                    pseudo_labels = y[idx_to_query]
                    # y[idx_to_query] = pseudo_labels
                print(f"New Label Counts: {Counter(y[select_ind[:batch_size]])}")    
                label_ind = np.concatenate([label_ind, select_ind[:batch_size]])# label_ind.update(select_ind)
                mask = np.where(np.isin(unlab_ind, select_ind[:batch_size], invert=True)) # unlab_ind.difference_update(select_ind)
                unlab_ind = unlab_ind[mask]
                
                print(f"Added {batch_size} Shape of Label_ind: {np.shape(label_ind)}")  
                #Update The Model
                X_scaled = self.scaler.transform(X.iloc[label_ind]) 
                model = self.model
                model.fit(X=X_scaled, y=y[label_ind]) 
            else:
                print("No Data Added")
            # informative_score_list.append(informative_score)
        merged_train_df = pd.DataFrame(X.iloc[label_ind], columns = features[2:])
        merged_train_df["Label"] = y[label_ind]
        new_train_df = merged_train_df.iloc[len(X_known):]
        # return new_train_df, informative_score_list, model
        return new_train_df, model
        
    def perform_inference(self, X):
            X_scaled = self.scaler.transform(X) # ! Add Scaler
            output = self.model.predict(X_scaled) 
            unique, counts = np.unique(output, return_counts=True)
            print(f"Malicious Request: {counts[0]} , Benign Request:{counts[1]}")
            return output
    def append_training_data(self,new_train_df):
        #Write a new CSV FIle
        folder = self.train_folder
        filepath = get_avail_filename(folder, "lm_train_data")
        if not new_train_df.empty:
            new_train_df.to_csv(filepath, index=False)
            print(f"Added {filepath} as New LM Training Data")
        else:
            print("New LM Training Data is Empty.. Skip Recording")
    def compress_training_data(self):
        pass
