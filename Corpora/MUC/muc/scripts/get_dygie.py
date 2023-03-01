import json
from nltk.tokenize import word_tokenize

input_prefix = "../processed/"
file_names = ["test.json", "dev.json", "train.json"]
output_prefix = "../processed-dygie/"

result = []
if __name__ == '__main__':
    for name in file_names:
        f = open(input_prefix + name)
        lines = f.readlines()
        f.close()
        for line in lines:
            gtt_format = json.loads(line)
            dygie_format = {'doc_key': gtt_format['docid'], 'sentences': [word_tokenize(gtt_format['doctext'])],
                            'dataset': "MUC3/4"}
            for template in gtt_format['templates']:
                for k,v in template:
                    if "ner" not in dygie_format:
                        dygie_format['ner'] = []
                    if k != "incident_type":
                        role_fillers = word_tokenize()

                        dygie_format['ner'].append()
            result.append(dygie_format)
