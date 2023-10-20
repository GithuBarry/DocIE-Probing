import json
import os

import nltk

path_gtt_style_with_trigger = ["../gtt_format/dev.jsonl", "../gtt_format/train.jsonl", "../gtt_format/test.jsonl"]
dygie_event_path = "../dygie_format/"

with open("../gtt_format/EventTypeToRoles.json", 'r') as f:
    event_to_roles = json.load(f)


def find_word_index(text, word, char_i):
    for c in range(-1, 2):
        while text.startswith(" "):
            c += 1
            text = text[1:]
        if text[char_i + c:].replace(" ", "").startswith(word.replace(" ", "")):
            return char_i + c

    index = len(nltk.casual_tokenize(text[:char_i])) - 1
    assert word.replace(" ", "") in "".join(nltk.casual_tokenize(text)[index:index + len(nltk.casual_tokenize(word))])
    return index


def get_trigger_ner_tag(incident_type):
    return incident_type + "_Trigger"


def process_file(file):
    with open(file, 'r') as f:
        gtt_examples = [json.loads(line) for line in f]

    event_file = []

    for example in gtt_examples:
        full_casual_tokenized = nltk.casual_tokenize(example['doctext'])
        all_ners_d = {}

        events = []
        for template in example['templates']:
            trigger = template.get("Trigger")
            incident_type = template['incident_type']
            trigger_char_index = template.get('TriggerIndex')
            trigger_word_index = find_word_index(example['doctext'], trigger, trigger_char_index)

            all_ners_d[(trigger_word_index, trigger_word_index)] = [trigger_word_index, trigger_word_index,
                                                                    get_trigger_ner_tag(incident_type)]

            event = [[trigger_word_index, get_trigger_ner_tag(incident_type)]]
            for key in event_to_roles[incident_type]:
                for mentions in template[key]:
                    index = find_word_index(example['doctext'], mentions[0][0], mentions[0][1])
                    end_index = index + len(nltk.casual_tokenize(mentions[0][0])) - 1
                    event.append([index, end_index, key])
                    all_ners_d[(index, end_index)] = [index, end_index, key]
            events.append(event)

        event_file.append({"doc_key": example['docid'], "dataset": "WikiEvents", "ner": [list(all_ners_d.values())],
                           "sentences": [full_casual_tokenized], "events": [events]})

    with open(os.path.join(dygie_event_path, os.path.basename(file)), "w+") as event_f:
        for item in event_file:
            event_f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    for file in path_gtt_style_with_trigger:
        process_file(file)
