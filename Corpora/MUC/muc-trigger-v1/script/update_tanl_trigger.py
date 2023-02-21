import json
import os

new_triggered_gtt_muc1700_path = ""
tanl_old_mucevent_path = "../../muc-trigger-v0"
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
                id = template['TriggerIndex'] if "TriggerIndex" in template else templates_dict[doc['docid']][
                    'doctext'].index(
                    template['Trigger'])

                trigger_len = len(template['Trigger'])

                assert templates_dict[doc['docid']]['doctext'][id:id + trigger_len] == template['Trigger']

                file[i]['triggers'].append({"type": template['incident_type'], "start": id, "end": id + 1})

        fp = open(os.path.join(tanl_new_mucevent_path, file_suffix), "w+")
        json.dump(file, fp)

        print("Finished")
