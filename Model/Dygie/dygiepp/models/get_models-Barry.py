import os
import shutil
import tarfile
import sys

epochs = [5,10,15,20]  # Specify the epoch numbers for which you want to create folders
seeds = [int(sys.argv[1])]

for seed in seeds:
    print("----Running on seed:",seed)
    os.chdir(f"/home/zw545/DocIE-Probing/Model/Dygie/dygiepp/models/muc_event_w_ner_fullep20seed{seed}")

    def compress_folder_to_tar_gz(folder_path, output_path):
        # Create the output tar.gz file
        with tarfile.open(output_path, "w:gz") as tar:
            # Iterate over the files and subdirectories in the folder
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Get the absolute path of the file
                    file_path = os.path.join(root, file)
                    # Calculate the relative path within the tar.gz file
                    relative_path = os.path.relpath(file_path, folder_path)
                    # Add the file to the tar.gz archive with the relative path
                    tar.add(file_path, arcname=relative_path)

    # Iterate over each epoch
    for epoch in epochs:
        epoch_folder = f"model_epoch_{epoch}"
        os.makedirs(epoch_folder, exist_ok=True)  # Create the epoch folder if it doesn't exist

        # Copy vocabulary folder to the epoch folder
        shutil.copytree("vocabulary", os.path.join(epoch_folder, "vocabulary"))

        # Copy model_state_epoch_x.th and rename it to weights.th
        model_state_file = f"model_state_epoch_{epoch-1}.th"
        weights_file = "weights.th"
        shutil.copy(model_state_file, os.path.join(epoch_folder, weights_file))

        # Copy config.json to the epoch folder
        shutil.copy("config.json", epoch_folder)

        # Create a tar.gz file for the epoch folder
        tar_file = f"{epoch_folder}.tar.gz"
        compress_folder_to_tar_gz(epoch_folder, tar_file)


        # Remove the uncompressed epoch folder
        shutil.rmtree(epoch_folder)

        print(f"Compressed {epoch_folder} into {tar_file}")
