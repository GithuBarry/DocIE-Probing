import json

file_names = ["mucevent_test.json", "mucevent_dev.json", "mucevent_train.json"]
output_file_name = "muc_1700_v0_GTT_style_triggered-test-dev-train.json"

result = []
for name in file_names:
    f = open(name)
    result.extend(json.load(f))
    f.close()

json.dump(result, open(output_file_name, "w"))
