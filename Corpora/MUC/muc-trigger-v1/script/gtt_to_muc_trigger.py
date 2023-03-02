var = {"doc_key": "02567fd428a675ca91a0c6786f47f3e35881bcbd-0", "dataset": "SciREX",
       "sentences": [["document", "vision"]], "ner": [
        [[3542, 3544, "Material"], [863, 865, "Metric"], [55, 56, "Task"], [3481, 3481, "Material"],
         [863, 865, "Metric"], [55, 56, "Task"]]]}

path_gtt_style_muc_with_trigger = "../muc_1700_v1.1.1_GTT_style_triggered-test-dev-train.json"
output_file1_name = "../muc_dygie/muc_trigger"
output_file2_name = "../muc_dygie/muc_event"


if __name__ == '__main__':
    f = open(path_gtt_style_muc_with_trigger)
    lines = f.readlines()
    trigger_file = []


