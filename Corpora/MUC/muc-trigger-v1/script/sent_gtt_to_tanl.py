# by Wayne Chen
import json

import nltk
from tqdm import tqdm


def search_sublst(sublst, lst, start=0):
    return_lst = []
    for idx in range(start, len(lst) - len(sublst) + 1):
        if lst[idx: idx + len(sublst)] == sublst:
            return_lst.append(idx)
    return return_lst


if __name__ == '__main__':
    examples = json.load(open("../sentmuc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"))
    write_lst = []
    for example in tqdm(examples):
        dictionary = example
        new_dictionary = {"docid": dictionary["docid"], "triggers": [], "entities": [], "relations": [],
                          "tokens": nltk.word_tokenize(dictionary["doctext"])}

        for template in dictionary["templates"]:
            trig_index = template["TriggerIndex"] if "TriggerIndex" in template else dictionary['doctext'].index(
                template["Trigger"])
            while trig_index < len(dictionary['doctext']) and str.isalpha(dictionary['doctext'][trig_index]):
                trig_index += 1
            trig_index = len(nltk.word_tokenize(dictionary['doctext'][:trig_index])) - 1
            assert template["Trigger"].split(" ")[0] in new_dictionary["tokens"][trig_index]

            new_dictionary["triggers"].append(
                {"type": template["incident_type"], "start": trig_index,
                 "end": trig_index + len(nltk.word_tokenize(template["Trigger"]))})

            for k in template.keys():
                if k not in ["incident_type", "Trigger", "ManuallyLabeledTrigger", "TriggerVersion",
                             "TriggerIndex"] and len(
                    template[k]) > 0:
                    for entities in template[k]:
                        for coref_list in entities:
                            span_as_tok = nltk.word_tokenize(coref_list[0])
                            ind = search_sublst(span_as_tok, new_dictionary["tokens"])

                            for i in ind:
                                entity_entry = {"type": k, "start": i, "end": i + len(span_as_tok)}
                                matched = False
                                for entity_ind, entity in enumerate(new_dictionary["entities"]):
                                    if entity == entity_entry:
                                        new_dictionary["relations"].append(
                                            {"type": k + ": " + template["incident_type"],
                                             "head": entity_ind,
                                             "tail": len(new_dictionary["triggers"]) - 1})
                                        matched = True
                                        break
                                if not matched:
                                    new_dictionary["entities"].append(entity_entry)
                                    new_dictionary["relations"].append({"type": k + ": " + template["incident_type"],
                                                                        "head": len(new_dictionary["entities"]) - 1,
                                                                        "tail": len(new_dictionary["triggers"]) - 1})

        write_lst.append(new_dictionary)
    for file, num in [("test", 200), ("dev", 200), ("train", 1300), ("test", 24049)]:
        with open("mucevent_{}.json".format(file), "w") as new_f:
            new_f.write(json.dumps(write_lst[:num]))
            write_lst = write_lst[num:]
    assert not write_lst
