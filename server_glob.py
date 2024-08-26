import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler
from deepod.models import PReNet

from model_lib import *

# Initialize Flask app
app = Flask(__name__)
# Initialize Global Model
clf = Global_Model()

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

# TODO LM Manager (Offload to other LM)


@app.route("/retrain", methods=['POST'])
# TODO: Retrain the model based on the updated

@app.route("/offload", methods=['POST'])
def detect():
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
    validated_flow_data = model_lib.validated_req_schema(flow_df)    
    validated_flow_data, origin_ip_series = get_user_ip(validated_flow_data.copy())
    # TODO: Append to training_csv

    
    isMalicious = clf.perform_inference(validated_flow_data)
    ip_label_tuple = list(zip(origin_ip_series.values, isMalicious))
    isMalicious_list = [int(x) for x in isMalicious]
    
    

    # Return the result as a JSON response
    return jsonify({
        "origin_ip": list(origin_ip_series),
        "Label": isMalicious_list
    }), 200
  

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 5050, threaded=True, debug=True)