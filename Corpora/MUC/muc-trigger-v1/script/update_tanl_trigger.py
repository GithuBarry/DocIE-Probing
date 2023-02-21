import json
import os

import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()

new_triggered_gtt_muc1700_path = "../muc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"
tanl_old_mucevent_path = "../../muc-trigger-v0/TANL_Style/"
tanl_new_mucevent_path = "../TANL_Style_NEW"

if __name__ == '__main__':
    examples = json.load(open(new_triggered_gtt_muc1700_path))

    templates_dict = dict()

    for i, e in enumerate(examples):
        templates_dict[examples[i]['docid']] = examples[i]

    for file_suffix in ["mucevent_dev.json", "mucevent_test.json", "mucevent_train.json"]:
        fp = open(os.path.join(tanl_old_mucevent_path, file_suffix))
        file = json.load(fp)
        fp.close()
        for i, doc in enumerate(file):
            triggers = []
            file[i]['triggers'] = []
            for template in templates_dict[doc['docid']]['templates']:
                char_index = template['TriggerIndex'] if "TriggerIndex" in template else templates_dict[doc['docid']][
                    'doctext'].index(
                    template['Trigger'])

                trigger_len = len(template['Trigger'])

                assert templates_dict[doc['docid']]['doctext'][char_index:char_index + trigger_len] == template[
                    'Trigger']

                # Find the correct word index by converting all characters prior to the trigger to words
                word_index = len(nltk.tokenize.word_tokenize(templates_dict[doc['docid']]['doctext'][:char_index]))

                # if word index does not correspond to the right word, try surrounding words:
                if ps.stem(file[i]['tokens'][word_index]) != ps.stem(template['Trigger']):
                    fixed = False
                    for offset in [-3, -2, -1, 0, 1, 2, 3]:
                        if template['Trigger'] in file[i]['tokens'][word_index + offset]:
                            word_index += offset
                            fixed = True
                            break

                    if not fixed:
                        print("Unmatched", doc['docid'], file[i]['tokens'][word_index], "<-->", template['Trigger'],
                              file[i]['tokens'][word_index - 5:word_index + 5])

                file[i]['triggers'].append(
                    {"type": template['incident_type'], "start": word_index, "end": word_index + 1})

        fp = open(os.path.join(tanl_new_mucevent_path, file_suffix), "w")
        json.dump(file, fp)

    print("Finished")
