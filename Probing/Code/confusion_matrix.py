import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    results_path = "../Results-MLPwAttention-BucketedRun2/"
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
            train_acc = result['train_acc']
            val_acc = result['val_acc']
            test_acc = result['test_acc']
            epoch = result["actual_epoch"]

            txt = f"Val acc {val_acc:.2f}, Test acc {test_acc:.2f}, Train acc {train_acc:.2f}\nEpoch {epoch}"
            cm_display.plot()
            cm_display.ax_.set_title(txt)

            plt.savefig(results_path+file[:-5] + ".png")
