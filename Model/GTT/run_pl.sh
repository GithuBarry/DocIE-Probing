#!/usr/bin/env bash

export MAX_LENGTH_SRC=435
export MAX_LENGTH_TGT=75
export BERT_MODEL=bert-base-uncased

export BATCH_SIZE=1
export NUM_EPOCHS=18
export SEED=1

export OUTPUT_DIR_NAME=model_GTT_18
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
rm -r "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
#rm ../data/scirex/processed/cached*

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../":"${PYTHONPATH}"


export th=80
echo "=========================================================================================="
echo "                                           threshold (${th})                              "
echo "=========================================================================================="
python3 run_pl_gtt.py \
  --data_dir ../../../Corpora/MUC/muc/processed \
  --model_type bert \
  --model_name_or_path $BERT_MODEL \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_length_src $MAX_LENGTH_SRC \
  --max_seq_length_tgt $MAX_LENGTH_TGT \
  --num_train_epochs $NUM_EPOCHS \
  --train_batch_size $BATCH_SIZE \
  --eval_batch_size $BATCH_SIZE \
  --seed $SEED \
  --n_gpu 1 \
  --thresh $th \
  --do_predict \
  --do_train
#done
