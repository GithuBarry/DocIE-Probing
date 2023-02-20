import os
import numpy as np

if __name__ == "__main__":
  firsts = []
  lasts =  []

  for folder in ["/home/zw545/GTT_Probe/model_gtt/test_hidden_layers", "/home/zw545/GTT_Probe/model_gtt/dev_hidden_layers","/home/zw545/GTT_Probe/model_gtt/train_hidden_state"]:
    file_name = dict()
    for file in os.listdir(folder):
      
      file_name[int(file.split("_")[3])] = os.path.join(folder,file) 
    i = 0
    while i in file_name:
      n = np.load(file_name[i])
      firsts.append(n[0][0])
      lasts.append(n[1][0])
      i+=1
  np.save("X_first_hidden_layer_GTT_20Epoch_bert_muc1700",np.array(firsts))
  np.save("X_last_hidden_layer_GTT_20Epoch_bert_muc1700",np.array(lasts))
  print("Done")



