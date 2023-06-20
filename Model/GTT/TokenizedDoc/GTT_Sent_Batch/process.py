import json
import os
from collections import defaultdict

tokenized = defaultdict(lambda: defaultdict(list))
order = json.load(open("../../../../Corpora/MUC/muc/processed2/sentmuc1700_25749order.json"))
file_list = []
if __name__ == '__main__':
    for i in range(26):
        for file in os.listdir('.'):
            if file.startswith(f'batch_{i}_') :
                file_path = os.path.join('.', file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        file_list.extend(data.values())
    for i, datum in enumerate(file_list):
        datum = datum['src_tokens']
        o = order[i]
        if len(o) == 1:
            tokenized["Full"][o[0]]=(datum)
        else:
            l = len(datum)
            # Sentence embedding
            left = 7  # Remove muc event prefix
            right = l - 1  # Remove last [Sep token]
            if o[1] == 0:
                left = 0
            if i + 1 == len(order) or order[i + 1][0] != o[0]:
                right += 1
            tokenized["Sent"][o[0]].extend(datum[left:right])
    json.dump(list(tokenized["Sent"].values()),open("GTT_MUC1700_SentCatTokenized.json", "w+"))
    json.dump(list(tokenized["Full"].values()), open("GTT_MUC1700_FullTextTokenized.json", "w+"))
