# by Wayne Chen
import json
from collections import defaultdict


def search_sublst(sublst, lst, start=0):
    return_lst = []
    for idx in range(start, len(lst) - len(sublst) + 1):
        if lst[idx: idx + len(sublst)] == sublst:
            return_lst.append(idx)
    return return_lst


relation_names = set()
entity_names = set()
incident_type_to_relations = defaultdict(set)

if __name__ == '__main__':
    for file in ["train", "dev", "test"]:
        f = open(f"../raw/{file}.jsonl", "r")
        raw_dicts = [json.loads(line) for line in f.readlines()]
        f = open(f"../gtt_format/{file}.jsonl", "r")
        gtt_dicts = [json.loads(line) for line in f.readlines()]

        write_lst = []

        for gtt_dict, raw_dicts in zip(gtt_dicts, raw_dicts):
            new_dictionary = {"docid": gtt_dict["docid"], "triggers": [], "entities": [], "relations": [],
                              "tokens": raw_dicts['tokens']}
            entity_ids = {d['id']: d for d in raw_dicts["entity_mentions"]}
            for gtt_template, raw_template in zip(gtt_dict["templates"], raw_dicts['event_mentions']):
                new_dictionary["triggers"].append(
                    {"type": gtt_template["incident_type"], **raw_template['trigger']})
                entity_names.add(gtt_template["incident_type"])

                for argument in raw_template['arguments']:
                    k = argument['role']
                    entity_names.add(k)
                    entity_entry = {"type": k, "start": entity_ids[argument['entity_id']]['start'],
                                    "end": entity_ids[argument['entity_id']]['end']}
                    matched = False
                    relation_name = k + ": " + gtt_template["incident_type"]
                    relation_names.add(relation_name)
                    incident_type_to_relations[gtt_template["incident_type"]].add(relation_name)
                    for entity_ind, entity in enumerate(new_dictionary["entities"]):
                        if entity == entity_entry:
                            new_dictionary["relations"].append(
                                {"type": relation_name,
                                 "head": entity_ind,
                                 "tail": len(new_dictionary["triggers"]) - 1})
                            matched = True
                            break
                    if not matched:
                        new_dictionary["entities"].append(entity_entry)
                        new_dictionary["relations"].append(
                            {"type": relation_name,
                             "head": len(new_dictionary["entities"]) - 1,
                             "tail": len(new_dictionary["triggers"]) - 1})

            write_lst.append(new_dictionary)

        with open("wikievents_{}.json".format(file), "w") as new_f:
            new_f.write(json.dumps(write_lst))
    with open("wikievents_schema.json", "w") as f:
        incident_type_to_relations = {k: list(v) for k, v in incident_type_to_relations.items()}
        json.dump(incident_type_to_relations, f)
    with open("wikievents_types.json", "w") as f:
        json.dump({"relations": {relation_name: {"verbose": relation_name} for relation_name in relation_names},
                   "entities": {entity_name: {"verbose": entity_name} for entity_name in (entity_names)}}, f, )
