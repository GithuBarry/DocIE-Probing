import os
import numpy as np
import gc
from collections import defaultdict
from tqdm import tqdm

if __name__ == "__main__":
  results = defaultdict(list)
  for i in tqdm(range(1700)):
    gc.collect()
    file_name =os.path.join("../HiddenStates/epoch20_new",f"output_sentence{str(i)}_encoder_hidden_states.npy")
    all_layers = np.load(file_name)
    for ii in range(len(all_layers)):
      l = all_layers[ii]
      while len(l)==1:
        l = l[0]
      results[ii].append(l) 
  for layer_id in results:
    layer_name = str(layer_id) if layer_id != len(results) -1 else "last"
    xfile_name = f"X_TANL_layer_{layer_name}_t5-base_epoch20_muc1700"
    np.save(xfile_name,np.array(results[layer_id]))
  print("Done")



