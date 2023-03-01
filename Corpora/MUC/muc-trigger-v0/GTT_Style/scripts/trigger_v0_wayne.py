def search_sublst(sublst, lst, start = 0):
    return_lst = []
    for idx in range(start, len(lst) - len(sublst) + 1):
        if lst[idx: idx + len(sublst)] == sublst:
            return_lst.append(idx)
    return return_lst

with open("muc_triggers.json", "r") as f:
    trigger_dict = json.loads(f.read())

for file in ["dev", "test", "train"]:
    trigger_labeled = []
    f = open("{}.json".format(file), "r")
    
    for line in f.readlines():
        info = json.loads(line)
        new_templates = []
        used_triggers = []
        doc_tokens = nltk.word_tokenize(info["doctext"])
        
        for template in info["templates"]:
            event = template["incident_type"]

            if event in trigger_dict["default"]:
                trig_ind_s, trig_ind_e, trigger = -1, -1, ""
                candidates = trigger_dict["default"][event]
                if event in trigger_dict["best"]:
                    candidates = trigger_dict["best"][event] + candidates

                found_trig = False
                trig_ind = 0
                while not found_trig:
                    trig = candidates[trig_ind]
                    trig_tokens = nltk.word_tokenize(trig)
                    ind_lst = search_sublst(trig_tokens, doc_tokens)

                    if len(ind_lst):
                        for index in ind_lst:
                            entry = (trig, index)
                            if not entry in used_triggers:
                                used_triggers.append(entry)
                                template["trigger"] = [trig, index, index + len(trig_tokens)]
                                found_trig = True
                                break
                    
                    trig_ind += 1

                new_templates.append(template)
        
        trigger_labeled.append({'docid': info['docid'], 'doctext': info['doctext'], 'templates': new_templates})
    
    f.close()
    with open("{}_with_trig.txt".format(file), "w") as f:
        for info in trigger_labeled:
            f.write(json.dumps(info))
            f.write("\n")