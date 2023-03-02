# by Wayne Chen
import json
import nltk


def search_sublst(sublst, lst, start=0):
    return_lst = []
    for idx in range(start, len(lst) - len(sublst) + 1):
        if lst[idx: idx + len(sublst)] == sublst:
            return_lst.append(idx)
    return return_lst


for file in ["train", "dev", "test"]:
    f = open("{}_with_trig.txt".format(file), "r")

    write_lst = []
    for line in f.readlines():
        dictionary = json.loads(line)
        new_dictionary = {"docid": dictionary["docid"], "triggers": [], "entities": [], "relations": [],
                          "tokens": nltk.word_tokenize(dictionary["doctext"])}

        for template in dictionary["templates"]:
            new_dictionary["triggers"].append(
                {"type": template["incident_type"], "start": template["trigger"][1], "end": template["trigger"][2]})

            for k in template.keys():
                if k not in ["incident_type", "trigger"] and len(template[k]) > 0:
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

    with open("mucevent_{}.json".format(file), "w") as new_f:
        new_f.write(json.dumps(write_lst))
