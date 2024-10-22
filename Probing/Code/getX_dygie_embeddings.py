from collections import defaultdict
import json
import os 


import numpy as np

if __name__ == "__main__":
    model = "dygiepp"
    hidden_state_path = f"../../Model/Dygie/{model}/HiddenStates/Embeddings"
    meta_data = "bert-uncased_epoch21_muc1700"
    examples_layers = defaultdict(list)

    muc_1700_input = open("../../Corpora/MUC/muc/processed2/muc_1700_GTT_style-test-dev-train.json")
    muc_ids = [example['docid'] for example in json.load(muc_1700_input)]

    

    for muc_id in muc_ids:
        npy = np.load(os.path.join(hidden_state_path, muc_id+".npy"))
        for i in range(len(npy)):
            examples_layers[i].append(npy[i] if len(npy[i])>1 else npy[i][0])
    
    for i, layer_key in enumerate(examples_layers.keys()):
        layer_name = f"layer_{layer_key}"  if i <len(examples_layers.keys())-1 else f"layer_last"
        file_name = f"../X/X_{model}_{layer_name}_{meta_data}"
        try:
            nparray = np.array(examples_layers[layer_key])
            np.save(file_name, nparray)
        except:
            print("cannot save as npy because varying embedding length")
            np.savez(file_name, examples_layers[layer_key])

        
    