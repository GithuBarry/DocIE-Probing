import os

if __name__ == '__main__':
    results_path = "../Results-MLPwAttention-Bucketed/"
    for file in os.listdir(results_path):
        #####Filtering#####
        if "nhid" in file:
            continue
        ###################
        prefix = "probresult"
        if file[:len(prefix)] == prefix:
            if file[-4:] == ".png":
                file_path = os.path.join(results_path, file)
                new_path = file_path[:-4] + "_nhid200" + file_path[-4:]
                os.renames(file_path, new_path)
