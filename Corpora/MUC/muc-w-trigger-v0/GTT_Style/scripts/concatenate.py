import json

prefix = "../Barry s version/triggered-"
file_names = ["test.json", "dev.json", "train.json"]
output_file_name = "muc_1700_v0.1_GTT_style_triggered-test-dev-train.json"

result = []
for name in file_names:
    f = open(prefix+name)
    lines = f.readlines()
    f.close()
    for line in lines:
        result.extend(json.loads(line))

json.dump(result, open(output_file_name, "w"))
