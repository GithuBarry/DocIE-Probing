from collections import defaultdict
import json
import os 


import numpy as np

if __name__ == "__main__":
    model = "GTT"
    meta_data = "bert-uncased_epoch20"
    examples_layers = defaultdict(list)

    hidden_state_path = f"../../Model/{model}/HiddenStates"

    for file in ["test","dev","train"]:
        folder = file+"_hidden_states"
        for example in os.listdir(os.path.join(hidden_state_path, folder )):
            npy = np.load(os.path.join(hidden_state_path, folder, example))
            for i in range(len(npy)):
                examples_layers[i].append(npy[i] if len(npy[i])>1 else npy[i][0])
    
    for i, layer_key in enumerate(examples_layers.keys()):
        layer_name = f"layer_{layer_key}"  if i <len(examples_layers.keys())-1 else f"layer_last"
        file_name = f"../X/X_{model}_{layer_name}_{meta_data}_muc1700"
        try:
            nparray = np.array(examples_layers[layer_key])
            np.save(file_name, nparray)
        except:
            print("cannot save as npy because varying embedding length")
            np.savez(file_name, examples_layers[layer_key])

        
    