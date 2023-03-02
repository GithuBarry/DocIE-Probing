import json
import os.path

import nltk

# Example format
# var = {"doc_key": "02567fd428a675ca91a0c6786f47f3e35881bcbd-0", "dataset": "SciREX",
#       "sentences": [["document", "vision"]], "ner": [
#        [[3542, 3544, "Material"], [863, 865, "Metric"], [55, 56, "Task"], [3481, 3481, "Material"],
#         [863, 865, "Metric"], [55, 56, "Task"]]]}

path_gtt_style_muc_with_trigger = "../muc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"
muc_dygie_trigger_path = "../muc_dygie/muc_trigger"
muc_dygie_event_path = "../muc_dygie/muc_event"  # TODO

if __name__ == '__main__':
    f = open(path_gtt_style_muc_with_trigger)
    trigger_file = []
    gtt_examples = json.load(f)
    f.close()

    for example in gtt_examples:
        full_casual_tokenized = nltk.casual_tokenize(example['doctext'])
        triggers = []
        for template in example['templates']:
            index = template['TriggerIndex'] if 'TriggerIndex' in template else example['doctext'].index(
                template["Trigger"])
            space_index = index

            trigger_index_word_incidenttype = -1 + len(nltk.casual_tokenize(example['doctext'][:index + 1])), template[
                'Trigger'], \
                template['incident_type']
            assert template['Trigger'] in " ".join(full_casual_tokenized[
                                                   trigger_index_word_incidenttype[0]:trigger_index_word_incidenttype[
                                                                                          0] + len(
                                                       template['Trigger'].split())])
            triggers.append(trigger_index_word_incidenttype)

        trigger_file.append({"doc_key": example['docid'],
                             "dataset": "MUC34",
                             "sentences": [full_casual_tokenized],
                             "ner": [[
                                 [index,
                                  # dygie NER's is right hand inclusive hence -1
                                  -1 + index + len(nltk.casual_tokenize(word)),
                                  incident_type + "Trigger"]
                                 for
                                 (index, word, incident_type) in triggers]
                             ]})

    for num, file_suffix in [(200, "test.json"), (200, "dev.json"), (1300, "train.json")]:
        with open(os.path.join(muc_dygie_trigger_path, file_suffix), "w+") as f:
            while num > 0:
                num -= 1
                f.write(json.dumps(trigger_file.pop(0)))
                f.write("\n")
