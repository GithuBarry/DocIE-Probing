import os
import numpy as np
import gc
from collections import defaultdict
from tqdm import tqdm

if __name__ == "__main__":
  results = defaultdict(list)
  for i in tqdm(range(1700)):
    gc.collect()
    file_name =os.path.join("./HiddenStates",f"hiddenstates_{str(i)}_alllayers.npy")
    all_layers = np.load(file_name)
    for ii in range(len(all_layers)):
      results[ii].append(all_layers[ii][0][:433]) #Removing all target tokens. Src tokens scale up to only 433
  for layer_id in results:
    layer_name = str(layer_id) if layer_id != len(results) -1 else "last"
    xfile_name = f"X_GTT_layer_{layer_name}_bert-uncased_epoch18_muc1700"
    np.save(xfile_name,np.array(results[layer_id]))
  print("Done")



