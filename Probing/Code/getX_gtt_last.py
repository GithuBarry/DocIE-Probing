import json
import os 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    
    X = np.load("/Users/barry/X_last_hidden_layer_GTT_20Epoch_bert_muc1700.npy")
    X_sel = "/Users/barry/Library/Mobile Documents/com~apple~CloudDocs/Programming/TokenizerPlayground/muc_period_indices_gtt_tokenizer.json"
    new_X = []
    ls = json.load(open(X_sel)).values()
    length = max(ls, key=lambda x: len(x))
    for i, l in enumerate(ls):
        new_vec = [X[i][ii] for ii in l]

        new_X.append(np.array(new_vec))

    a = new_X
    b = np.zeros([len(a), len(max(a, key=lambda x: len(x))), 768])
    for i, j in enumerate(a):
        b[i][0:len(j)] = j

    X = np.array(b)
    X = X.reshape((len(X), -1))

    np.save("/Users/barry/X_last_hidden_layer_period_only_GTT_20Epoch_bert_muc1700.npy", X)
    