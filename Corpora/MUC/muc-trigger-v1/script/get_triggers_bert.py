# This script finds triggers in a GTT style MUC dataset

import json
from collections import Counter
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")
import torch
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)



old_triggered_muc1700_path = "../../muc-trigger-v0/GTT_Style/muc_1700_v0.1_GTT_style_triggered-test-dev-train.json"
new_muc1700_path = "muc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"

selected_trigger = {
    'kidnapping': ["kidnap", "kidnapping", "kidnapped", "abducted", "forced", "forces", "hostage", "hostages",
                   "capture", "capture", "arrested"],
    'attack': ["attack", "attacked", "attacks", "murder", "murdered", "kill", "killed", "forced", "force",
               "assassinated", "assassination", "assassinate", "death", "shot", "shooting", "violence", "raid",
               "massacre", "dead"],
    'bombing': ["bomb", "bombed", "explosion", "bombs", "exploded", "explode", "explodes", "explosive", "dynamite",
                "blew", "detonated", "detonates", "blown", "mined", "grenades", "grenade", "blowing-up"],
    'arson': ["fire", "burned", "burn"],
    'robbery': ["rob", "force", "robbed", "robbing", "forced", "force", "stole"],
    'forced work stoppage': ["strike", "stoppage"]
}
role_list = ['Victim', 'Target', 'PerpInd', 'PerpOrg', "Weapon"]


def find_all(a_str, sub):
    """
    Returns all occurrences
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def get_role_filler_positions(template):
    indices = []
    for role in role_list:
        for role_fillers in template[role]:
            for role_filler in role_fillers:
                mention = role_filler[0]
                index = role_filler[1]
                indices.append(index)
    return indices


if __name__ == "__main__":
    f = open(old_triggered_muc1700_path)

    count_repeated_trigger = set()
    count_has_emptied = []
    count_no_role_filler_indices = set()
    examples = json.load(f)
    f.close()

    templates_dict = dict()

    # Proved that the index data is auto generated (first occurrence) only

    for i, e in enumerate(examples):
        templates = e["templates"]
        text = e['doctext']
        types = [t["incident_type"] for t in templates]

        for (incident_type, freq) in Counter(types).most_common():

            if freq >= 2:  # Only deal with multi-template example
                # For each template that has a potentially shared trigger:
                selected_trigger_indices = []
                selected_trigger_e_word = []
                for ii, template in enumerate(e["templates"]):
                    if template['incident_type'] == incident_type:
                        # Heuristic to find the best trigger:
                        indices = get_role_filler_positions(template)
                        empty = len(indices) == 0
                        repeated_templates = -1

                        # Should there be two templates of the same type sharing same role fillers, rule out them in
                        # considering best trigger
                        for template2 in e["templates"]:
                            if template2['incident_type'] == incident_type and template2 != template:
                                for id in get_role_filler_positions(template2):
                                    while id in indices:
                                        indices.remove(id)
                                repeated_templates += 1
                        if not empty and len(indices) == 0:
                            count_has_emptied.append(i)
                            indices = get_role_filler_positions(template)

                        best_trigger = None if len(selected_trigger_e_word) == 0 else selected_trigger_e_word[-1]
                        best_distance = float('inf')
                        # Use glove vectors and cosine similarity for best distance
                        
                        best_index = None if not selected_trigger_indices else selected_trigger_indices[-1]

                        for trigger_candidate in selected_trigger[incident_type]:
                            if trigger_candidate in e['doctext']:
                                trigger_tokens = tokenizer.tokenize(trigger_candidate)
                                trigger_ids = tokenizer.convert_tokens_to_ids(trigger_tokens)
                                trigger_ids = torch.tensor(trigger_ids).unsqueeze(0)
                                with torch.no_grad():
                                    bert_output = model(trigger_ids)[0].squeeze(0)
                                trigger_vec = bert_output.detach().numpy()
                                trigger_indices = find_all(text, trigger_candidate)

                                distance = float('inf')
                                trigger_index_best = 0

                                for trigger_index in trigger_indices:
                                    # Choose best occurrence
                                    # print([e['doctext'][index] for index in indices])
                                    #indices are the role filler position
                                    sims = []
                                    for index in indices:
                                        filler_tokens = tokenizer.tokenize(text[index])
                                        filler_ids = tokenizer.convert_tokens_to_ids(filler_tokens)
                                        filler_ids = torch.tensor(filler_ids).unsqueeze(0)
                                        with torch.no_grad():
                                            bert_output = model(filler_ids)[0].squeeze(0)
                                        filler_vec = bert_output.detach().numpy()
                                        # get cosine similarity from bert
                                      
                                        sims.append(cosine_similarity(trigger_vec, filler_vec)[0][0])
                                    if sims:
                                        new_distance = sum(sims) / len(sims)
                                    else:
                                        new_distance = 0

                                    if repeated_templates <= 0 and trigger_index in selected_trigger_indices:
                                        new_distance = float('inf')
                                    if not indices:
                                        count_no_role_filler_indices.add(i)
                                    if distance > new_distance:
                                        distance = new_distance
                                        trigger_index_best = trigger_index

                                if distance < best_distance:
                                    # See if the best occurrence is better than other trigger candidate
                                    best_distance = distance
                                    best_trigger = trigger_candidate
                                    best_index = trigger_index_best

                        # print(best_distance)
                        # print(text[:best_index], "***", text[best_index:])
                        selected_trigger_indices.append(best_index)
                        selected_trigger_e_word.append(best_trigger)
                        examples[i]['templates'][ii]['Trigger'] = best_trigger
                        examples[i]['templates'][ii]['TriggerIndex'] = best_index
                        examples[i]['templates'][ii]['TriggerVersion'] = 'V1.1-UsingHeuristicForMultiTemplate'

                if len(set(selected_trigger_indices)) < len(selected_trigger_indices):
                    count_repeated_trigger.add(i)

                if None in selected_trigger_indices:
                    print("Bad")
    for i, e in enumerate(examples):
        templates_dict[examples[i]['docid']] = examples[i]

    print("count_repeated_trigger       ", len(count_repeated_trigger))
    print("count_no_role_filler_indices ", len(count_no_role_filler_indices))
    print("count_has_emptied            ", len(count_has_emptied))
    fp = open(new_muc1700_path, "w+")
    json.dump(examples, fp)
