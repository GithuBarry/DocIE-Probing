import json
import os
from collections import defaultdict

empty_template = {
    "incident_type": None,
    "PerpInd": [],
    "PerpOrg": [],
    "Target": [],
    "Victim": [],
    "Weapon": []
}


def run_with(raw_output_dir, muc_gtt_input_path, parsed_output_path):
    f = open(muc_gtt_input_path)
    gtt = {json.loads(line)['docid']: json.loads(line) for line in f.readlines()}
    f.close()

    if not os.path.exists(parsed_output_path):
        os.mkdir(parsed_output_path)

    results = {}
    for file in os.listdir(raw_output_dir):
        file_path = os.path.join(raw_output_dir, file)
        f = open(file_path)
        json_lines = f.readlines()
        f.close()

        for json_line in json_lines:
            output = json.loads(json_line)
            sentence = output['sentences'][0]
            gtt_input = gtt[output['doc_key']]

            # Extract templates
            templates = []
            for event in output['predicted_events'][0]:
                template = defaultdict(list)
                for argument in event:
                    if len(argument) == 4:
                        parsed_argument = sentence[argument[0]:argument[0] + 1]
                        role = argument[1]
                    else:
                        parsed_argument = sentence[argument[0]:argument[1] + 1]
                        role = argument[2]

                    if 'Trigger' in role:
                        template['incident_type'] = role[:-len('Trigger')]
                        #template['trigger'] = parsed_argument[0]
                        #template['trigger_index'] = argument[0]
                    else:
                        template[role].append([" ".join(parsed_argument)])
                        #template[role + "_index"].append([(argument[0], argument[1])])
                for key in empty_template.keys():
                    if key not in template:
                        template[key] = []

                if template['incident_type']:
                    templates.append(dict(template))
            results[output['doc_key']] = {'doctext': gtt_input['doctext'],
                                          'pred_templates': templates,
                                          'gold_templates': gtt_input['templates']
                                          }
        f = open(os.path.join(parsed_output_path, "formatted_" + file[:-1]), "w+")
        json.dump(results, f)
        f.close()


if __name__ == '__main__':
    files = ["test"]
    for file in files:
        raw_output_dir = "../Outputs/RawModelOutputs"
        muc_gtt_input_path = "../../../../Corpora/MUC/muc/processed/" + file + ".json"
        parsed_output_path = "../Outputs/FormattedModelOutputs"
        run_with(raw_output_dir=raw_output_dir, muc_gtt_input_path=muc_gtt_input_path,
                 parsed_output_path=parsed_output_path)
