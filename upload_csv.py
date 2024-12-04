import requests
import argparse
import pandas as pd
import pickle
from io import StringIO

def send_csv(url, file_path):
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        # Convert JSON response back into DataFrame
        json_data = response.json()
        df = pd.DataFrame(json_data, columns=["origin_ip", "Label"])
        
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

# def load_train_model(labeled_df):
#   model_location = "decision_tree_classifier.pkl"
#   with open(model_location, 'rb') as file:
#     CART = pickle.load(file)
#   model = CART  
#   y = labeled_df['Label']
#   X = labeled_df.drop(columns=["Label"])
  
#   model.fit(X=X,y=y)
#   with open(model_location,'wb') as file:
#     pickle.dump(model,file)
#     print("Model Trained and Updated")
  
      
def main():
    parser = argparse.ArgumentParser(description='Upload File to csv.')
    # parser.add_argument('url', type=str, help='The URL of the server to which the file is sent.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file to send.')
    
    args = parser.parse_args()
    url = "http://localhost:3001/detect"
    labeled_df = send_csv(url, args.file_path)
    labeled_df.to_csv("ip_labels.csv")
    # if isinstance(labeled_df, pd.DataFrame):
    #   load_train_model(labeled_df)
      

if __name__ == '__main__':
    main()