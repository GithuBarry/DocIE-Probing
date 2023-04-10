import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    for file in os.listdir():
        prefix = "result_"
        if file[:len(prefix)] == prefix and file[-5:] == ".json" and ("events" in file or "num_sent" in file):
            print(file)
            with open(file) as f:
                result = json.load(f)

            confusion_matrix = metrics.confusion_matrix(np.array(result["y_true"]), np.array(result["y_pred_int"]))
            print(confusion_matrix)

            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=sorted(
                list(set(result["y_true"] + result["y_pred_int"]))))

            cm_display.plot()
            plt.savefig(file[:-5] + ".png")
