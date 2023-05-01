import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    results_path = "../Results-MLPwAttention-Bucketed/"
    for file in os.listdir(results_path):
        #####Filtering#####
        if "nhid400" not in file:
            continue
        ###################

        prefix = "probresult"
        if file[:len(prefix)] == prefix and file[-5:] == ".json":
            print(os.path.join(results_path, file))
            with open(os.path.join(results_path, file)) as f:
                result = json.load(f)
            labels = np.array([l.index(max(l)) for l in result["val_true"]])
            preds = np.array([l.index(max(l)) for l in result["val_pred"]])
            confusion_matrix = metrics.confusion_matrix(labels, preds)
            print(confusion_matrix)

            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                        display_labels=sorted(list(set(list(labels) + list(preds)))))

            cm_display.plot()
            plt.savefig(file[:-5] + ".png")
