import json

import nltk

# This file expand muc1700 to muc1700+ per-sentence with empty template result.
# Useful when getting sentence embedding
input_file_name = "../processed2/muc_1700_GTT_style-test-dev-train.json"
output_file_name = "../processed2/sentmuc_1700_GTT_style-test-dev-train.json"
output_jsonl_name = "../processed2/sentmuc_1700_GTT_style-test-dev-train.jsonl"
output_id_file_name = "../processed2/sentmuc_order.json"

input_data = json.load(open(input_file_name))

input_data_len = len(input_data)
if __name__ == '__main__':
    new_examples = []
    new_examples_id = []
    for example in input_data:
        text = example['doctext']
        sents = nltk.sent_tokenize(text)
        for id, sent in enumerate(sents):
            id_str = str(id)
            new_examples.append({
                "docid": example['docid'] + "#" + id_str,
                "doctext": sent,
                "templates": []
            })
            new_examples_id.append([example['docid'], id])

    output = input_data + new_examples
    json.dump(output, open(output_file_name, "w+"))
    json.dump([[ex["docid"]] for ex in input_data] + new_examples_id, open(output_id_file_name, "w+"), )

    with open(output_jsonl_name, 'w') as outfile:
        for entry in output:
            json.dump(entry, outfile)
            outfile.write('\n')
