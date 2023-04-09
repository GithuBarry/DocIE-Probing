import matplotlib.pyplot as plt

import json

import os
import numpy as np
from sklearn import metrics

for file in os.listdir():
    prefix = "result_"
    if file[:len(prefix)] == prefix and file[-5:] == ".json" and "num_token" not in file:
        print(file)
        with open(file) as f:
            result = json.load(f)
        
        confusion_matrix = metrics.confusion_matrix(np.array(result["y_true"]),np.array( result["y_pred_int"]))
        print(confusion_matrix)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels= sorted(list(set(result["y_true"]+result["y_pred_int"]))))

        cm_display.plot()
        plt.savefig(file[:-5]+".png")