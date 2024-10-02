import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBClassifier
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler
from model_lib import *
import requests
import threading

# Initialize Flask app
app = Flask(__name__)
# def load_model():
# model = joblib.load('cic_xgb.joblib')
# scaler = joblib.load('cic_scaler.joblib')
lm = Local_Model() # ! Change GM to API
def decode_json(request):
  data = request.json
  return data

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
  
def offload_to_gm(url, validated_flow_data):
    flow_json_data = validated_flow_data.to_json(orient="records") 
    response = requests.post(url, json=flow_json_data)
    if response.status_code == 200:
        # Convert JSON response back into DataFrame
        json_data = response.json()
        df = pd.DataFrame(json_data, columns=["origin_ip", "Label"])
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
def retrain_model_async(X_new, y_pseudo, threshold):
    # Asynchronous retraining function
    lm.retrain_model(X_new, y_new=y_pseudo, threshold=threshold)    
    
@app.route("/retrain", methods=['POST'])
def retrain():
    # TODO: While state = retrain, offload to other models
    validated_flow_data = validated_req_schema(request, is_labeled=True)
    validated_flow_data.to_csv("x.csv", index=False)
    X_new = validated_flow_data.drop(columns=["Label", "Src IP", "Dst IP"], inplace=False)
    y_new = validated_flow_data["Label"]
    retrain_thread = threading.Thread(target=retrain_model_async, args=(X_new,y_new, 0.05))
    retrain_thread.start()
    return jsonify({'message': "Retraining Started Successfully"}, 200)
    

  
@app.route("/offload", methods=['POST'])
def offload():

  # TODO: Maybe LM Receive Data from the network monitor and the labels from the GM (Avoid High GM Uplink) || Use time for mapping data to labels
  # ! But for now, data sent to LM then forwarded
  offload_url = request.form.get('offload_url') # !Get URL From API Request
  print(offload_url)
  # Send Data to GM
  validated_flow_data = validated_req_schema(request) #Get the Model URL from LMM
  
  label_df = offload_to_gm(offload_url, validated_flow_data)
  # Record The Data
  validated_flow_data["Label"] = label_df["Label"] # ! Check this guy
  validated_flow_data.to_csv("x.csv")
  
  # Retrain
  X_new = validated_flow_data.drop(columns=["Label", "Src IP", "Dst IP"], inplace=False)
  y_pseudo = validated_flow_data["Label"]
  
  # Start retraining in a separate thread
  retrain_thread = threading.Thread(target=retrain_model_async, args=(X_new, y_pseudo, 0.05))
  retrain_thread.start()  
  
  # TODO: Train after returning
  # Return the results
  validated_flow_data.to_csv("x.csv")
  _, origin_ip_series = get_user_ip(validated_flow_data[["Src IP", "Dst IP", "Label"]].copy())
  isMalicious_list = np.array(y_pseudo.values).astype(int).tolist()
  
  return jsonify({
      "origin_ip": list(origin_ip_series),
      "Label": isMalicious_list
  }), 200

@app.route("/detect", methods=['POST'])
def detect():

    validated_flow_data = validated_req_schema(request)
    validated_flow_data.to_csv("x.csv")
    validated_flow_data, origin_ip_series = get_user_ip(validated_flow_data.copy())

    # ! Can't find user_ip

    
    isMalicious = lm.perform_inference(validated_flow_data)
    ip_label_tuple = list(zip(origin_ip_series.values, isMalicious))
    # ip_malic_df = pd.DataFrame(ip_label_tuple, columns=["origin_ip", "Labels"])
    # ip_malic_df.to_csv('ip_malic.csv')

    # Convert isMalicious to a list of native Python types
    isMalicious_list = np.array(isMalicious).astype(int).tolist()

    # Return the result as a JSON response
    return jsonify({
        "origin_ip": list(origin_ip_series),
        "Label": isMalicious_list
    }), 200
    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 3001, threaded=True, debug=True)