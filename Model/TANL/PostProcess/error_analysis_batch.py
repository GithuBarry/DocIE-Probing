import os
import subprocess

if __name__ == '__main__':
    model_output_folder = "../Outputs"
    analysis_output_folder = os.path.join(model_output_folder, "ErrorAnalysisResult")
    if not os.path.exists(analysis_output_folder):
        os.mkdir(analysis_output_folder)
    ps = []
    for file in os.listdir(model_output_folder):
        if file[-5:] == ".json" and "nonempty" not in file:
            model_out_path = os.path.join(model_output_folder, file)
            analysis_output_json = os.path.join(analysis_output_folder, "Analyzed_" + file)
            analysis_output_out = os.path.join(analysis_output_folder, "Analyzed_" + file[:-5] + ".out")
            ps.append(subprocess.Popen(
                ["python", "../../../Error_Analysis.py",
                 "-i", model_out_path,
                 "-j", analysis_output_json,
                 "-o", analysis_output_out,
                 "-s", "all",
                 "-m", "MUC_Errors"]))
    for p in ps:
        p.wait()
