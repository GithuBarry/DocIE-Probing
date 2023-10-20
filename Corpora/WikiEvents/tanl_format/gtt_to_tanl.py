# by Wayne Chen
import json


def search_sublst(sublst, lst, start=0):
    return_lst = []
    for idx in range(start, len(lst) - len(sublst) + 1):
        if lst[idx: idx + len(sublst)] == sublst:
            return_lst.append(idx)
    return return_lst


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

                for argument in raw_template['arguments']:
                    k = argument['role']
                    entity_entry = {"type": k, "start": entity_ids[argument['entity_id']]['start'],
                                    "end": entity_ids[argument['entity_id']]['end']}
                    matched = False
                    for entity_ind, entity in enumerate(new_dictionary["entities"]):
                        if entity == entity_entry:
                            new_dictionary["relations"].append(
                                {"type": k + ": " + gtt_template["incident_type"],
                                 "head": entity_ind,
                                 "tail": len(new_dictionary["triggers"]) - 1})
                            matched = True
                            break
                    if not matched:
                        new_dictionary["entities"].append(entity_entry)
                        new_dictionary["relations"].append(
                            {"type": k + ": " + gtt_template["incident_type"],
                             "head": len(new_dictionary["entities"]) - 1,
                             "tail": len(new_dictionary["triggers"]) - 1})

            write_lst.append(new_dictionary)

        with open("{}.json".format(file), "w") as new_f:
            new_f.write(json.dumps(write_lst))
