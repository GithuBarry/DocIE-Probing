import json
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    dataset_name = "wikievents246"
    muc_1700_input = open("../../Corpora/WikiEvents/gtt_format/all-test-dev-train.jsonl")
    examples = [json.loads(l) for l in muc_1700_input.readlines()]
    muc_1700_input.close()

    label_sets = {"num_events": [len(example['templates']) for example in examples],
                  "num_words": [len(nltk.word_tokenize(example['doctext'])) for example in examples],
                  "num_sent": [len(nltk.sent_tokenize(example['doctext'])) for example in examples]
                  }

    data = dict()
    for key in label_sets:
        labels = label_sets[key]
        np.save(f"Y_{key}_{dataset_name}", np.array(labels))

        # Using negative number to prioritize bucketing long tail.
        # Otherwise
        cuts = pd.qcut([-l for l in labels], q=10, duplicates='drop').categories
        labels_bucketed_10 = [len(cuts) - [-nt in cut for cut in cuts].index(True) - 1 for nt in labels]

        enc = OneHotEncoder()
        reshaped_array = np.array(labels_bucketed_10).reshape(-1, 1)
        labels_bucketed_10_onehot = enc.fit_transform(reshaped_array).toarray()
        np.save(f"Y_bucket_{key}_{dataset_name}", np.array(labels_bucketed_10_onehot))

        print(key)
        data[key] = dict(Counter(labels))
        print(data[key])
        data[key + "bucket"] = dict(Counter(labels_bucketed_10))
        print(data[key + "bucket"])

    json.dump(data, open(f"./label_stats_{dataset_name}.json", "w+"))
