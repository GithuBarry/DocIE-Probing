import json
import os.path
from tqdm import tqdm
import nltk
from pathlib import Path

# Example format
# var = {"doc_key": "02567fd428a675ca91a0c6786f47f3e35881bcbd-0", "dataset": "SciREX",
#       "sentences": [["document", "vision"]], "ner": [
#        [[3542, 3544, "Material"], [863, 865, "Metric"], [55, 56, "Task"], [3481, 3481, "Material"],
#         [863, 865, "Metric"], [55, 56, "Task"]]]}

path_gtt_style_muc_with_trigger = "../sentmuc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"
muc_dygie_trigger_path = "../sentmuc_dygie/muc_trigger"
muc_dygie_event_path = "../sentmuc_dygie/muc_event_w_ner"
Path(muc_dygie_trigger_path).mkdir(parents=True, exist_ok=True)
Path(muc_dygie_event_path).mkdir(parents=True, exist_ok=True)

def find_word_index(text, word, char_i):
    index = char_i
    index = -1 + len(nltk.casual_tokenize(text[:index + 1]))
    word_len = len(nltk.casual_tokenize(word))
    full_casual_tokenized = nltk.casual_tokenize(text)
    assert word.replace(" ", "") in "".join(full_casual_tokenized[index:index + word_len])
    return index


def get_trigger_ner_tag(incident_type: str) -> str:
    return incident_type + "Trigger"


if __name__ == '__main__':
    f = open(path_gtt_style_muc_with_trigger)
    trigger_file = []
    event_file = []
    gtt_examples = json.load(f)
    f.close()

    for example in tqdm(gtt_examples):
        full_casual_tokenized = nltk.casual_tokenize(example['doctext'])
        triggers = []
        events = []
        all_ners_d = {}
        doc_text = example['doctext']
        for template in example['templates']:
            trigger = template['Trigger']
            incident_type = template['incident_type']
            trigger_char_index = template['TriggerIndex'] if 'TriggerIndex' in template else doc_text.index(
                template["Trigger"])
            trigger_word_index = find_word_index(doc_text, trigger, trigger_char_index)

            trigger_index_word_incidenttype = (trigger_word_index, trigger, incident_type)

            triggers.append(trigger_index_word_incidenttype)
            all_ners_d[(trigger_word_index, trigger_word_index)] = [trigger_word_index,
                                                                    trigger_word_index,
                                                                    get_trigger_ner_tag(incident_type)]
            event = [[trigger_word_index, incident_type + "Trigger"]]
            for key in ['PerpInd', 'PerpOrg', 'Target', 'Victim', 'Weapon']:
                for mentions in template[key]:
                    index = find_word_index(doc_text, mentions[0][0], mentions[0][1])
                    end_index = -1 + index + len(nltk.casual_tokenize(mentions[0][0]))
                    event.append([index, end_index, key])
                    all_ners_d[(index, end_index)] = [index, end_index, key]

            events.append(event)

        trigger_file.append({"doc_key": example['docid'],
                             "dataset": "MUC34",
                             "sentences": [full_casual_tokenized],
                             "ner": [[
                                 [index,
                                  # dygie NER's is right hand inclusive hence -1
                                  -1 + index + len(nltk.casual_tokenize(word)),
                                  get_trigger_ner_tag(incident_type)]
                                 for
                                 (index, word, incident_type) in triggers]
                             ]})
        event_file.append({"doc_key": example['docid'],
                           "dataset": "MUC34",
                           "ner": [list(all_ners_d.values())],
                           "sentences": [full_casual_tokenized],
                           "events": [events]})

    for num, file_suffix in [(200, "test.json"), (200, "dev.json"), (1300, "train.json"), (24049, "test.json")]:
        with open(os.path.join(muc_dygie_trigger_path, file_suffix), "w+") as trigger_f:
            with open(os.path.join(muc_dygie_event_path, file_suffix), "w+") as event_f:
                while num > 0:
                    num -= 1
                    trigger_f.write(json.dumps(trigger_file.pop(0)))
                    trigger_f.write("\n")
                    event_f.write(json.dumps(event_file.pop(0)))
                    event_f.write("\n")
    assert len(trigger_file) == 0
    assert len(event_file) == 0
