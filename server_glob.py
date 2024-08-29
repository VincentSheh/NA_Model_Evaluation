import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import joblib
from deepod.models import PReNet

from model_lib import *

# Initialize Flask app
app = Flask(__name__)
# Initialize Global Model
gm = Global_Model(['dripper/', 'BENIGN/', 'bonesi/'])

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

@app.route("/retrain", methods=['POST'])
# TODO: Retrain the model based on the updated
def retrain():
    validated_flow_data = validated_req_schema(request) #!NOT IMPLEMENTED
    gm.retrain_gm(validated_flow_data)
    # gm.load_model()

@app.route("/offload", methods=['POST'])
def rcv_offload():

    validated_flow_data = validated_req_schema(request, is_json=True)    
    validated_flow_data, origin_ip_series = get_user_ip(validated_flow_data.copy())

    isMalicious = gm.perform_inference(validated_flow_data)
    ip_label_tuple = list(zip(origin_ip_series.values, isMalicious))
    
    # Convert isMalicious to a list of native Python types
    isMalicious_list = np.array(isMalicious).astype(int).tolist()

    # Return the result as a JSON response
    return jsonify({
        "origin_ip": list(origin_ip_series),
        "Label": isMalicious_list
    }), 200
    
@app.route("/detect", methods=['POST'])
def detect():
    
    validated_flow_data = validated_req_schema(request, is_json=False)    
    validated_flow_data, origin_ip_series = get_user_ip(validated_flow_data.copy())

    isMalicious = gm.perform_inference(validated_flow_data)
    ip_label_tuple = list(zip(origin_ip_series.values, isMalicious))
    
    # Convert isMalicious to a list of native Python types
    isMalicious_list = np.array(isMalicious).astype(int).tolist()

    # Return the result as a JSON response
    return jsonify({
        "origin_ip": list(origin_ip_series),
        "Label": isMalicious_list
    }), 200    
  

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 5050, threaded=True, debug=False)