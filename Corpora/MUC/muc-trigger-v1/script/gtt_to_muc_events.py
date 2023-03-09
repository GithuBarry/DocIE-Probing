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


def find_word_index(text, word, char_i):
    index = char_i
    index = -1 + len(nltk.casual_tokenize(text[:index + 1]))
    word_len = len(nltk.casual_tokenize(word))
    full_casual_tokenized = nltk.casual_tokenize(text)
    assert word.replace(" ", "") in "".join(full_casual_tokenized[index:index + word_len])
    return index


if __name__ == '__main__':
    f = open(path_gtt_style_muc_with_trigger)
    trigger_file = []
    event_file = []
    gtt_examples = json.load(f)
    f.close()

    for example in gtt_examples:
        full_casual_tokenized = nltk.casual_tokenize(example['doctext'])
        triggers = []
        events = []
        for template in example['templates']:
            index = find_word_index(example['doctext'], template['Trigger'],
                                    template['TriggerIndex'] if 'TriggerIndex' in template else example[
                                        'doctext'].index(
                                        template["Trigger"]))
            incident_type = template['incident_type']

            trigger_index_word_incidenttype = index, \
                template['Trigger'], \
                incident_type

            triggers.append(trigger_index_word_incidenttype)
            event = [[index, incident_type + "Trigger"]]
            for key in ['PerpInd', 'PerpOrg', 'Target', 'Victim', 'Weapon']:
                for mentions in template[key]:
                    index = find_word_index(example['doctext'], mentions[0][0], mentions[0][1])
                    event.append([index, -1 + index + len(nltk.casual_tokenize(mentions[0][0])), key])

            events.append(event)

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
        event_file.append({"doc_key": example['docid'],
                           "dataset": "MUC34",
                           "ner": [[]],
                           "sentences": [full_casual_tokenized],
                           "events": [events]})

    for num, file_suffix in [(200, "test.json"), (200, "dev.json"), (1300, "train.json")]:
        with open(os.path.join(muc_dygie_trigger_path, file_suffix), "w+") as trigger_f:
            with open(os.path.join(muc_dygie_event_path, file_suffix), "w+") as event_f:
                while num > 0:
                    num -= 1
                    trigger_f.write(json.dumps(trigger_file.pop(0)))
                    trigger_f.write("\n")
                    event_f.write(json.dumps(event_file.pop(0)))
                    event_f.write("\n")
