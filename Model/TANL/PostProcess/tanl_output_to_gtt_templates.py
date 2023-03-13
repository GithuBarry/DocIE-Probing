import ast
import json
import re
from collections import defaultdict
from typing import List

import nltk

output_path = "./sample_out.out"
muc_test_path = "../data/mucevent/mucevent_test_only.json"
save_path = "output.json"
save_path_non_empty = "output_non_empty.json"

f = open(output_path)
lines = f.readlines()
f.close()

f = open(muc_test_path)
test_file = json.load(f)
f.close()

trigger_predict_prefix = "trigger_output_sentence"  # Printing of this indicates a start of a new example
gold_template_prefix = " gt_relations"  # Note there is a space. Could be multiple per example. Ignorable.
pred_template_prefix = "predicted_relations"  # Could be multiple per example

doc_index = -201

"""
"pred_templates": [
            {
                "incident_type": "attack",
                "PerpInd": [],
                "PerpOrg": [
                    [
                        "military"
                    ],
                    [
                        "armed forces"
                    ]
                ],
                "Target": [],
                "Victim": [],
                "Weapon": []
            }
        ]
"""
trigger_sample = "trigger_output_sentence today , medellin , colombia 's second largest city , once again experienced a terrorist escalation when seven bank branch offices were shaken by explosives that caused heavy damage but no fatalities , according to radio reports broadcast in bogota ( 500 km to the south ) . the targets of the [ attacks | attack ] were the banco cafetero branches and its offices in medellin 's middle , western , and southeastern areas . according to preliminary reports , over 55 kg of [ dynamite | bombing ] were used in the attacks . the radio report added that the police defused another 20 kg of explosives that had slow burning fuses . the medellin cartel operates in this city located in colombia 's northeastern area . for several days now , the city has been shaken by army and police operations in an unprecedented action to capture drug lords . no one has claimed responsibility for the terrorist attacks , which lasted for 1 hour ."
nltk.data.path.append('/Users/barry/.nltk_data')


def extract_trigger(s: str, org_text: List[str]):
    tokens = org_text
    regex_str = r"\[ [a-z]* \| [a-z]* \]"
    splits_re = re.split(regex_str, s)
    triggers_re = re.findall(regex_str, s)
    # index = 0
    if len(triggers_re) == 0:
        return []
    triggers = []
    assert len(splits_re) - 1 == len(triggers_re)
    for i, no_trigger_part in enumerate(splits_re):
        if i == len(triggers_re):
            break
        # index += len(nltk.word_tokenize(no_trigger_part))
        trigger = triggers_re[i].split("|")[0][1:].strip()
        inc_type = triggers_re[i].split("|")[1][:-1].strip()
        # triggers.append((index, trigger, inc_type))
        triggers.append((trigger, inc_type))
        # assert tokens[index] in trigger
        # index += len(nltk.word_tokenize(trigger))
    return triggers


id_to_templates = defaultdict(lambda: defaultdict(list))

if __name__ == '__main__':
    cur_triggers = []
    for line in lines:
        if line[:len(trigger_predict_prefix)] == trigger_predict_prefix:
            doc_index += 1
            if doc_index < 0:
                continue
            if doc_index >= 200:  # MUC test only has 200 examples.
                break
            cur_tokens = test_file[doc_index]['tokens']
            id_to_templates[doc_index]['pred_trigger_in_text'] = line[len(trigger_predict_prefix) + 1:]
            cur_triggers = extract_trigger(line[len(trigger_predict_prefix) + 1:], cur_tokens)
            id_to_templates[doc_index]['pred_triggers'] = cur_triggers
            for k, v in test_file[doc_index].items():
                id_to_templates[doc_index][k] = v
            template_id = -1

        empty_template = {
            "incident_type": None,
            "PerpInd": [],
            "PerpOrg": [],
            "Target": [],
            "Victim": [],
            "Weapon": []
        }
        if doc_index > 0 and line[:len(pred_template_prefix)] == pred_template_prefix:
            template_id += 1
            template_str = line[len(pred_template_prefix):]
            template = empty_template
            if "set()" in template_str:
                template['incident_type'] = cur_triggers[template_id][1]
                id_to_templates[doc_index]['pred_relations'].append([])
                if len(cur_triggers) != 1:
                    print("Must manually check doc_index for empty template's incident type:",
                          doc_index,
                          cur_triggers)
            else:
                parsed = ast.literal_eval("[" + template_str[2:-2] + "]")  # Assuming each line is of form " {(..)}\n"
                id_to_templates[doc_index]['pred_relations'].append(parsed)
                for argument in parsed:
                    role = argument[0].split(":")[0].strip()
                    template['incident_type'] = argument[0].split(":")[1].strip()
                    template[role].append(cur_tokens[argument[1][0]:argument[1][1]])

            id_to_templates[doc_index]['pred_templates'].append(template)

    result_dict = dict()
    result_dict_non_empty = dict()
    for index, value in id_to_templates.items():
        result_dict[index] = dict(value)
        if value['pred_templates']:
            result_dict_non_empty[index] = dict(value)


    json.dump(result_dict, open(save_path, "w+"))
    json.dump(result_dict_non_empty, open(save_path_non_empty, "w+"))
