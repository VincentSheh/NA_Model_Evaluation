import requests
import argparse
import pandas as pd
import pickle
from io import StringIO
import numpy as np
import os
def send_csv(url, file_path, offload_url = "http://localhost:5050/", action="offload"):
    files = {'file': open(file_path, 'rb')}
    data = {'offload_url': offload_url}
    print(url+action)
    response = requests.post(url+action, files=files, data=data)
    if response.status_code == 200:
        if action == 'retrain': #Doesn't Return Dataframe
            print(f"\033[32mRetraining Success\033[0m")
            return None
        # Convert JSON response back into DataFrame
        json_data = response.json()
        df = pd.DataFrame(json_data, columns=["origin_ip", "Label"])
        
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

def merge_csv(file_path):
    directory = os.path.dirname(file_path)
    path_list = os.listdir(directory)
    path_list = [file for file in path_list if file.startswith('merged')]
    full_df = pd.DataFrame()
    for file in path_list:
        full_file_path = os.path.join(directory, file)
        curr_file_df = pd.read_csv(full_file_path)
        if not curr_file_df.empty:
            full_df = pd.concat([curr_file_df,full_df], axis=0, ignore_index=True)
    full_df.to_csv(os.path.join(directory,"final_merged_3.csv"),index=False)
    print(f"Merged CSV Files {path_list} in {directory}")

def main():
    parser = argparse.ArgumentParser(description='Upload File to CSV.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file to send.')
    parser.add_argument('action', type=str, help='Action={detect, retrain, offload}')
    
    args = parser.parse_args()
    url = "http://localhost:5050/"

    try:
        labeled_df = send_csv(url, args.file_path, action=args.action)
        if labeled_df is not None and not labeled_df.empty:
            labeled_df.to_csv("ip_labels.csv", index=False)
            malicious_ip = labeled_df.loc[labeled_df["Label"] == 1, "origin_ip"]
            print(np.unique(malicious_ip))
            
            if len(malicious_ip) > 0:
                for ip in np.unique(malicious_ip):
                    print(ip)
                    # print(f"Blocked {ip}")
                    # os.system(f"sudo iptables -A INPUT -s {ip} -j DROP")
                    # os.system("sudo iptables -L INPUT --line-numbers | wc -l")
            else:
                print("No Malicious Traffic Detected")
            # merge_csv(args.file_path)
        else:
            print("No data returned from the server or the DataFrame is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Merge All CSV File in the directory
    # merge_csv(args.file_path)

if __name__ == '__main__':
    main()
    #sudo /home/vs/miniconda3/bin/python upload_csv.py /home/vs/Desktop/Network_Anomaly/packet_collector/cicflowmeter/attack/merged_20240809091837_ISCX.csv