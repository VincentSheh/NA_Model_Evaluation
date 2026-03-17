
# %config IPCompleter.greedy=True
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from xgboost import XGBClassifier
import random

import matplotlib as matplot
import matplotlib.pyplot as plt
# %matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings, os 
from matplotlib import pyplot as plt

from sklearn.metrics import *
from sklearn.model_selection import train_test_split


from deepod.models import PReNet
from collections import Counter
# from modAL.models impor ActiveLearner, Committee
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from itertools import combinations, product

LIGHT_GREEN = '\033[92m'  # Light green
GREEN = '\033[32m'  # Green
BLUE = '\033[34m'  # Green
PURPLE = '\033[35m'  # Green

RESET = '\033[0m'   # Reset to default color

dataset_csv_path = './Dataset/CICIDS2017_improved'
csv_file_names = os.listdir(dataset_csv_path)[:]
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
def read_csv():
    complete_paths = []
    for csv_file_name in csv_file_names:
        complete_paths.append(os.path.join(dataset_csv_path, csv_file_name))

    improved_df = pd.concat(map(pd.read_csv, complete_paths), 
                            ignore_index = True)
    dropping_cols = ['Protocol', 'id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 
                    'Dst Port', 'Timestamp']    
    improved_df = clean_df(improved_df)
    improved_df.drop(dropping_cols, axis = 1, inplace = True)
    improved_df['Label'].value_counts()    
    return improved_df

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

def preprocess_data(df, train_perc=0.15, random_state=42):
    # Make a copy of the dataframe
    improved_df = df.copy()
    
    # Clean the labels
    #* Remove Attack data labeled as Benign
    # attepmted_labels = [s for s in improved_df['Label'].unique() if 'Attempted' in s]
    # improved_df.drop(['Attempted Category'], axis=1, inplace=True)
    # improved_df.replace(attepmted_labels, 'BENIGN', inplace=True)
    #* --

    # Sample the anomaly rate
    # print(f"improved_df.shape {improved_df.shape} Before Sampling Out")
    # improved_df = sample_df(improved_df, anomaly_rate)
    # print(f"improved_df.shape {improved_df.shape} after Sampling Out")
    # ratio = improved_df.loc[improved_df["Label"] == "BENIGN"].shape[0] / improved_df.loc[improved_df["Label"] != "BENIGN"].shape[0]
    # print(f"Ratio of Benign to Anomaly is {ratio:.2f} : 1")
    
    # Prepare features and labels
    scaler = StandardScaler()
    Y = improved_df['Label'].map({"BENIGN": 0}).fillna(1)  # Map BENIGN to 0 and others to 1
    X = improved_df.drop(columns=["Label"], axis=1)
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, 
        Y, 
        test_size=1-train_perc, 
        shuffle=True, 
        stratify=Y, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def balance_attack(df, atk_list, verbose=0):
    atk_list = ["Portscan", "DoS Hulk", "DDoS", "Infiltration - Portscan", "DoS GoldenEye", "FTP-Patator", "DoS Slowloris", "SSH-Patator", "DoS Slowhttptest"]
    reduced_cicids_df = pd.DataFrame()
    for atk_name in atk_list:
        temp_df = df.loc[df["Label"] == atk_name]
        temp_df = temp_df.sample(n=min(temp_df.shape[0], 10_000))
        reduced_cicids_df = pd.concat([reduced_cicids_df, temp_df], ignore_index=True)
    
    benign_df = df.loc[df["Label"] == 'BENIGN']
    reduced_cicids_df = pd.concat([reduced_cicids_df, benign_df], ignore_index=True)
        
    if verbose:
        print(reduced_cicids_df["Label"].value_counts())
    return reduced_cicids_df
    
    
        
#* Usage
# read_csv() => balance_attack => define_atk / preprocess_data() =>Train()
atk_list = ["Portscan", "DoS Hulk", "DDoS", "Infiltration - Portscan", "DoS GoldenEye", "FTP-Patator", "DoS Slowloris", "SSH-Patator", "DoS Slowhttptest"]
cicids_df = read_csv()
reduced_cicids_df = balance_attack(cicids_df, atk_list, verbose=1)
cicids_df = reduced_cicids_df.copy()
# X_gm_train, X_gm_test, y_gm_train, y_gm_test = preprocess_data(reduced_cicids_df, train_perc=0.99, anomaly_rate)

benign_df = cicids_df.loc[cicids_df["Label"] == "BENIGN"]
benign_df = benign_df.sample(frac=1).reset_index(drop=True)
benign_df
def set_supervised_label(supervision_rate, y, idx):
        if supervision_rate == 0:
            y[idx[:2]] = 1
        else:
            idx = np.random.choice(idx, size=int(supervision_rate * len(idx)), replace=False)
            y[idx] = 1
        return y
      
def get_optimal_threshold(precision, recall, thresholds):
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0  # Replace NaN values with 0    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold  
      
class Global_Model():
  def __init__(self, supervision=1.0, initial_data=None):
      self.initial_data = initial_data.copy() if initial_data else None #Initially Empty
      self.gm_data = None
      self.supervision = supervision
      self.knowledge_pool = []
    #   self.gm_retrain(supervision=1.0)


  def gm_retrain(self, new_gm_data=None, share_knowledge=False):
      if new_gm_data is None:
          training_data = self.initial_data
      else:
          # Combine new data with existing GM data
          self.gm_data = pd.concat([self.gm_data, new_gm_data], ignore_index=True)
          training_data = pd.concat([self.initial_data, self.gm_data], ignore_index=False)
      
    #   print("GM Training Data Size", training_data.shape)
      label = training_data["Label"]
      training_data = training_data.drop(columns=["Label"])
      # Train-test split
      X_train, X_test, y_train, y_test = train_test_split(
          training_data, label, shuffle=True, test_size=0.2, random_state=4022
      )
      # Prepare labels for semi-supervised learning
      idx = np.where(y_train == 1)[0]
    #   print("BEFORE",idx.shape, y_train.shape)
      y_train = np.zeros_like(y_train)  # Initialize with zeros
      y_train = set_supervised_label(self.supervision, y_train, idx)
    #   print("AFTER", np.where(y_train == 1)[0])

      # Train the model
      model = PReNet
      self.clf = model(device="cuda", verbose=1, epochs=1, batch_size=64)
      self.clf.fit(X_train.to_numpy(), y=y_train)
      
      # Calculate optimal threshold
      anomaly_scores = self.clf.decision_function(X_test.to_numpy())
      precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
      opt_threshold = get_optimal_threshold(precision, recall, thresholds)
      self.opt_threshold = opt_threshold      

      # Perform inference to evaluate the model
      output, anomaly_scores = self.perform_inference(X_test, y_test)
      if share_knowledge:
        self.share_knowledge(new_gm_data)
    # return clf, model
              
  def perform_inference(self, X, y_real=None):  
      anomaly_scores = self.clf.decision_function(X.to_numpy())
      output = np.where(anomaly_scores > self.opt_threshold, 1,0)
      
      if y_real is not None:
          print("Accuracy of GM: ", accuracy_score(output, y_real))
      return output, anomaly_scores
  def share_knowledge(self, X_new):
    # X_new["Label"] = y_new
    for neigh_area in self.knowledge_pool:
        neigh_model = neigh_area.model
        neigh_model.gm_retrain(X_new, share_knowledge=False)
        # neigh_model.gm_data = pd.concat([neigh_model.lm_data, X_new.copy()], ignore_index=True)
        # neigh_model.model.fit(neigh_model.lm_data.drop(columns=["Label"]), neigh_model.lm_data["Label"])  





class Local_Model():
    def __init__(self, gm):
        self.model = XGBClassifier(objective='binary:logistic', verbosity=0)
        # self.model = DecisionTreeClassifier(criterion='entropy', max_depth=5,  
        #                                     min_samples_leaf=10, 
        #                                     # ccp_alpha=0.01, #Pruning coef
        #                                     random_state=4022)
        self.global_model = gm 
        self.lm_data = pd.DataFrame()
        self.knowledge_pool = []
                
    def retrain_model(self, new_lm_data, threshold = 0.2, update_gm = False): #Select Most Important Data and Upload Newly Recorded Data
        # TODO: Use AL to Select Prerecorded Data
        # known_df = self.load_known_df()
        # filtered_new_data, informative_score_list, updated_model = self.select_data(known_df, X_new, threshold, y_new)
        # labeled_new_data = self.upload_gm(filtered_new_data)
        
        # if update_gm:
        #     filtered_new_data["Label"] = 0
        #     self.global_model.update_data(filtered_new_data)
        # # TODO: After updating return the labels or recall the function
        # else:
        
        y_real = new_lm_data["Label"].copy()
        X_new = new_lm_data.drop(columns=["Label"]).copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_real, shuffle=True, test_size=0.2, random_state=4022
        )

        # Use the global model to inquire labels for X_train
        y_new = self.inquire_gm(X_train, y_train)

        # Combine X_train and the inferred labels into a DataFrame
        new_training_data = X_train.copy()
        new_training_data["Label"] = y_new

        # Append new training data to the local model's data
        self.lm_data = pd.concat([self.lm_data, new_training_data], ignore_index=True)

        # Retrain the local model using the updated data
        self.model.fit(self.lm_data.drop(columns=["Label"]), self.lm_data["Label"])

        # Evaluate the updated model (optional, based on your implementation)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        self.share_knowledge(X_train, y_new)

        return acc
    

        
        
    def inquire_gm(self, X_query, y_real=None): 
        # X_query_scaled = self.scaler.transform(X_query)
        pseudo_label, _ = self.global_model.perform_inference(X_query, y_real)
        X_query_df = pd.DataFrame(X_query)
        # X_query_df["Label"] = pseudo_label
        return pseudo_label
    
    def perform_inference(self, X):
        output = self.model.predict(X) 
        return output    
    


class Attack:
    def __init__(self, name, centroid, std_dev, data):
        self.name = name
        self.centroid = centroid
        self.std_dev = std_dev
        self.data, self.test_data = train_test_split(data, train_size=0.8, random_state=42)
        self.active = False
        self.rr_counter = 0  # Round Robin Counter
        self.first_seen = random.uniform(0,30)

    def get_data(self):
        chunk_size = self.data.shape[0] // 20  # Divide into 10 chunks
        start_idx = self.rr_counter * chunk_size
        end_idx = start_idx + chunk_size
        if self.rr_counter == 9:
            end_idx = self.data.shape[0]  # Include all remaining rows
        self.rr_counter = (self.rr_counter + 1) % 20
        return self.data.iloc[start_idx:end_idx]
        

def define_edge_area(num_edge_area, gm=None, random_state=42, supervision = 1.0, global_sharing=False):
    random.seed(random_state)
    area_list = [Edge_Area((random.uniform(0, 1), random.uniform(0, 1)), Global_Model()) for _ in range(num_edge_area)]
    model = Global_Model(supervision=supervision)
    for edge_area in area_list:
        centroid = edge_area.centroid
        distances = [
            np.sqrt((other_area.centroid[0] - centroid[0])**2 + (other_area.centroid[1] - centroid[1])**2)
            for other_area in area_list
        ]
        # Get the indices of the two smallest distances, excluding itself
        closest_indices = np.argsort(distances)[1:3]  # Skip the first (distance to itself)
        if gm is not None:
            # edge_area.model.knowledge_pool = [area_list[i] for i in closest_indices] #?Cluster KS
            edge_area.model.knowledge_pool = [] #? No KS
            #* Global Knowledge Sharing
            if global_sharing:
                # local_model = Local_Model(gm) 
                # edge_area.local_model.knowledge_pool = []
                edge_area.model = model
    return area_list

def define_atk(atk_list, edge_area_list, cicids_df, benign_df, random_state=42):
    random.seed(random_state)
    atk_info_dict = {}
    anomaly_ratio = 25
    benign_idx = 0
    for idx, atk in enumerate(atk_list):
        # Cycle through edge areas
        edge_area = edge_area_list[idx % len(edge_area_list)]

        # Define standard deviations
        std_dev = (random.uniform(0.2, 0.4), random.uniform(0.2, 0.4))

        # Offset the attack centroid relative to the edge area centroid
        atk_centroid = (
            edge_area.centroid[0] + random.uniform(-std_dev[0], std_dev[0]) / 3,
            edge_area.centroid[1] + random.uniform(-std_dev[1], std_dev[1]) / 3
        )

        # Filter attack-specific data
        atk_data = cicids_df.loc[cicids_df["Label"] == atk]

        # Get corresponding benign data based on anomaly ratio
        benign_count = atk_data.shape[0] * anomaly_ratio
        benign_data = benign_df.iloc[benign_idx: benign_idx + benign_count]
        benign_idx += benign_count

        # Ensure benign_idx doesn't exceed benign_df bounds
        if benign_idx > len(benign_df):
            benign_idx = len(benign_df)  # Prevent index out of bounds
            print("Index Out of Bounds", atk)

        # Concatenate benign and attack data
        data = pd.concat([benign_data, atk_data], ignore_index=True)
        data = data.sample(frac=1).reset_index(drop=True)
        # data["Label"] = data["Label"].map({"BENIGN": 0}).fillna(1)

        # Create and store an Attack object
        atk_info_dict[atk] = Attack(atk, atk_centroid, std_dev, data)

    return atk_info_dict


class Edge_Area:
    def __init__(self, centroid, gm=None):
        self.centroid = centroid
        self.seen_attack_acc = dict() # Values as List of (t, acc)
        if gm != None:
            self.model = gm
    def create_knowledge_sharing_pool(self):
        # TODO: Get the closest based on euclidean distance
        self.sharing_pool = None

from scipy.stats import norm

def probability_in_2d_edge_area(mu_x, sigma_x, mu_y, sigma_y, area_centroid_x, area_centroid_y):
    (a_x, b_x) = (area_centroid_x-0.1, area_centroid_x+0.1)
    (a_y, b_y) = (area_centroid_y-0.1, area_centroid_y+0.1)
    
    # Check if the centroid is more than 2 std away in either dimension
    if abs(mu_x - area_centroid_x) > 2 * sigma_x or abs(mu_y - area_centroid_y) > 2 * sigma_y:
        return 0.0
    
    # Probability in x-dimension
    p_x = norm.cdf(b_x, loc=mu_x, scale=sigma_x) - norm.cdf(a_x, loc=mu_x, scale=sigma_x)
    # Probability in y-dimension
    p_y = norm.cdf(b_y, loc=mu_y, scale=sigma_y) - norm.cdf(a_y, loc=mu_y, scale=sigma_y)

    # Total probability in the 2D region
    return p_x * p_y

# Example usage
mu_x, sigma_x = 0.5, 0.1  # Mean and std deviation for x
mu_y, sigma_y = 0.5, 0.1  # Mean and std deviation for y
area_centroid_x, area_centroid_y = (0.4, 0.4)  # Edge area bounds for x and y

probability = probability_in_2d_edge_area(mu_x, sigma_x, mu_y, sigma_y, area_centroid_x, area_centroid_y)
print(f"Probability of attack occurring in the edge area: {probability:.4f}")



num_runs = 1
num_timesteps = 1000
num_edge_area = 5
supervision = 0.05
global_sharing = False
cluster_sharing = True


from sklearn.exceptions import NotFittedError
import gc

# Main simulation function
final_accuracies = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    run=run+1

    # Initialize Global Model
    # global_model = Global_Model(gm_training_data[:100000])
    
    # Assign edge area coordinates
    edge_area_list = define_edge_area(5, gm="aaa", random_state=run, supervision = supervision, global_sharing=global_sharing)

    # Define attack information
    atk_info_list = define_atk(atk_list, edge_area_list, cicids_df, benign_df, random_state=run)

    # Loop through timesteps
    for t in range(num_timesteps):
        for attack in atk_info_list.values():
            # Sample whether an attack is ongoing
            for i, edge_area in enumerate(edge_area_list):
                occ_prob = probability_in_2d_edge_area(attack.centroid[0], attack.std_dev[0], 
                                                        attack.centroid[1], attack.std_dev[1], 
                                                        edge_area.centroid[0], edge_area.centroid[1])
                attack.active = np.random.random() < occ_prob/3
                if attack.active and attack.first_seen < t:
                    # Train local model with a fraction of the data     
                    X_lm_train, X_lm_test, y_lm_train, y_lm_test = preprocess_data(attack.get_data().copy(), train_perc=0.8)                   
                    print(y_lm_train.value_counts(), y_lm_test.value_counts())
                    y_lm_train = y_lm_train.values
                    y_lm_test = y_lm_test.values
                    # Record Accuracy Data
                    X_atk_test, _, y_atk_test, _ = preprocess_data(attack.get_data().copy(), train_perc=0.99)                   
                    y_atk_test = y_atk_test.values
                    
                  

                    try:
                        # Evaluate GM on LM test set
                        y_gm_pred = edge_area.model.perform_inference(X_lm_test)[0]
                        gm_acc = accuracy_score(y_lm_test, y_gm_pred)
                        gm_f1 = f1_score(y_lm_test, y_gm_pred, average='binary')

                        tn, fp, fn, tp = confusion_matrix(y_lm_test, y_gm_pred).ravel()
                        gm_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        gm_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                        # Evaluate GM on ATK test set
                        y_gm_atk_pred = edge_area.model.perform_inference(X_atk_test)[0]
                        gm_atk_acc = accuracy_score(y_atk_test, y_gm_atk_pred)
                        gm_atk_f1 = f1_score(y_atk_test, y_gm_atk_pred, average='binary')

                        tn, fp, fn, tp = confusion_matrix(y_atk_test, y_gm_atk_pred).ravel()
                        gm_atk_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        gm_atk_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0                             

                    except AttributeError as e:
                        # Handle the case where the model is not yet fitted
                        print("Local model is not fitted yet. Please fit the model before inference.")
                        gm_acc = 0
                        gm_f1 = 0
                        gm_fpr = 0
                        gm_fnr = 0
                        
                        gm_atk_acc = 0
                        gm_atk_f1 = 0
                        gm_atk_fpr = 0
                        gm_atk_fnr = 0
                    print(f"Timestep {t}, Area {i} Attack {attack.name}")
                    print(f"{LIGHT_GREEN}Accuracy of GM Before: {gm_acc} {gm_f1} {gm_fpr} {gm_fnr} || {gm_atk_acc} {gm_atk_f1} {gm_atk_fpr} {gm_atk_fnr} {RESET} ")                            
                    
                    # Update seen_attack_acc
                    if attack.name not in edge_area.seen_attack_acc:
                        edge_area.seen_attack_acc[attack.name] = [(t, gm_atk_acc, gm_atk_f1)]
                    else:
                        edge_area.seen_attack_acc[attack.name].append((t, gm_atk_acc, gm_atk_f1))
                        
                    if gm_f1 < 0.96: #* GM is the bottleneck Prev = 0.96
                        X_lm_train["Label"] = y_lm_train
                        # print("GM New Data Size", X_lm_train.shape[0])
                        edge_area.model.gm_retrain(X_lm_train, share_knowledge=cluster_sharing)
                        y_gm_pred = edge_area.model.perform_inference(X_lm_test)[0]
                        gm_acc = accuracy_score(y_gm_pred, y_lm_test)
                        gm_f1 = f1_score(y_gm_pred, y_lm_test, average='binary')
                        
                        tn, fp, fn, tp = confusion_matrix(y_gm_pred, y_lm_test).ravel()
                        gm_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        gm_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0                            
                        
                        y_gm_atk_pred = edge_area.model.perform_inference(X_atk_test)[0]
                        gm_atk_acc = accuracy_score(y_gm_atk_pred, y_atk_test)
                        gm_atk_f1 = f1_score(y_gm_atk_pred, y_atk_test, average='binary')
                        tn, fp, fn, tp = confusion_matrix(y_gm_atk_pred, y_atk_test).ravel()
                        gm_atk_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        gm_atk_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                        print(f"{BLUE}Accuracy of GM After: {gm_acc} {gm_f1} {gm_fpr} {gm_fnr} || {gm_atk_acc} {gm_atk_f1} {gm_atk_fpr} {gm_atk_fnr} {RESET} ")
                    print(attack.name, X_lm_train.shape)
                    gc.collect()
                    

