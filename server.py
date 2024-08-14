import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBClassifier
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)
# def load_model():
model = joblib.load('cic_xgb.joblib')
scaler = joblib.load('cic_scaler.joblib')
def decode_json(request):
  data = request.json
  return data
def validated_req_schema(flow_data):
  # pruned_features=['Init_Win_bytes_forward', 'Bwd Packets/s',
  #      'Init_Win_bytes_backward', 'Flow Duration', 'Packet Length Std',
  #      'PSH Flag Count', #TODO
  #      'Fwd Packets/s', #TODO
  #     #  'Destination Port', 
  #     #  'min_seg_size_forward', #TODO
  #      'Average Packet Size', 
  #      'Total Length of Bwd Packets', 'Bwd Packet Length Min',
  #      'Fwd IAT Min', #TODO
  #      'Fwd Header Length', 'Total Backward Packets',
  #      'Total Length of Fwd Packets', 
  #      'Bwd Packet Length Mean',
  #      'Bwd Header Length', #TODO
  #     #  'Packet Length Mean', #TODO SIMILAR To Average Packet Size?
  #      'Flow IAT Min']
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
  #
  # df['Destination Port'] = df['Destination Port'].astype(int)
  # Select only the required columns
  df_pruned = flow_data[features]
      
  return df_pruned  
def get_user_ip(X):
    known_ip = {"192.168.50.12"}
    # Determine the origin IP
    X["origin_ip"] = X.apply(lambda x: x['Src IP'] if x['Dst IP'] in known_ip else x['Dst IP'], axis=1)
    # Delete Inf Entries
    X.replace([np.inf, -np.inf], np.nan, inplace = True)
    X.dropna(inplace=True)
    # Extract the origin_ip column and drop unnecessary columns in one step
    origin_ip_series = X["origin_ip"].copy()
    X = X.drop(columns=['Src IP', 'Dst IP', 'origin_ip'])
    print(origin_ip_series.values)
    return X, origin_ip_series


def perform_inference_sup(X):
  X.to_csv("x.csv")

  X_scaled = scaler.transform(X)
  output = model.predict(X_scaled) 
  print(np.unique(output)) 
  return output




@app.route("/detect", methods=['POST'])
def detect():
    # flow_data = decode_json(request) # ? Inference Using JSON

    # csv_dest = "global.csv" # ? Save File
    # Check if a file is part of the POST request
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
    validated_flow_data = validated_req_schema(flow_df)    
    validated_flow_data, origin_ip_series = get_user_ip(validated_flow_data.copy())

    # ! Can't find user_ip

    
    isMalicious = perform_inference_sup(validated_flow_data)
    ip_label_tuple = list(zip(origin_ip_series.values, isMalicious))
    # ip_malic_df = pd.DataFrame(ip_label_tuple, columns=["origin_ip", "Labels"])
    # ip_malic_df.to_csv('ip_malic.csv')

    # Convert isMalicious to a list of native Python types
    isMalicious_list = [int(x) for x in isMalicious]

    # Return the result as a JSON response
    return jsonify({
        "origin_ip": list(origin_ip_series),
        "Label": isMalicious_list
    }), 200
  

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 3001, threaded=True)