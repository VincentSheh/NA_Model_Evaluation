from model_lib import *
import torch
if __name__ == '__main__':
  gm = Global_Model(['BENIGN/', 'bonesi/'])
  # print("A")
  X_train_ge, X_test_ge, y_train_ge, y_test_ge = load_data(train_folder=["BENIGN/"])
  counter = Counter(y_train_ge)
  counter
  torch.cuda.empty_cache()
  scaler = joblib.load('./cic_scaler.joblib')
  X_test_scaled_ge = scaler.transform(X_test_ge)
  gm_out = gm.perform_inference(X_test_ge)
  print("Accuracy of GM", accuracy_score(gm_out,y_test_ge), "F1_Score", f1_score(gm_out, y_test_ge))
