#!/bin/bash
#SBATCH --job-name=run_scirex_both
#SBATCH --open-mode=append
#SBATCH --output=/home/ks4765/fa21_ie/%j_%x.out
#SBATCH --error=/home/ks4765/fa21_ie/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:40:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8


## activate your Conda environment in the container before running your Python script.
singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
conda activate dygiepp
bash scripts/train.sh scirex 
"

allennlp predict models/promed-TEST/model.tar.gz \
    data/promed/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/promedTEST.jsonl \
    --cuda-device 0 \
    --silent