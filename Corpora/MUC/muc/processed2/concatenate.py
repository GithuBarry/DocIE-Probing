import json

prefix = "../processed/"
file_names = ["test.json", "dev.json", "train.json"]
output_file_name = "muc_1700_GTT_style-test-dev-train.json"

result = []
for name in file_names:
    f = open(prefix+name)
    lines = f.readlines()
    f.close()
    for line in lines:
        result.append(json.loads(line))

json.dump(result, open(output_file_name, "w"))
