# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 13:04:56 2021

@author: DS
"""


import Runner
BASE_DIR=r"../InputsAndOutputs/"

BERT_PATH = BASE_DIR+"pretrained/CODER"
TOKENIZER_PATH= BERT_PATH
MODEL_TYPE="syn_spert"
CONFIG_PATH= BASE_DIR + "configs/config-coder.json"

LOG_PATH=  BASE_DIR + "data/log/"
SAVE_PATH= BASE_DIR + "data/save/"
CACHE_PATH= BASE_DIR + "data/cache/"

# Paths to data files for training, validation, and testing
TRAIN_PATH = "your/train/path/md_train_KG_0217_1_agu_new.json"  # Set your train file path here
VALID_PATH = "your/valid/path/md_test_KG_0217_1_agu_new.json"  # Set your validation file path here
TYPES_PATH = BASE_DIR + "data/datasets/DPKG_types_Cor4.json"  # Set your types file path here (BASE_DIR should be defined)
TEST_PATH = "your/test/path/md_test_KG_0216_agu.json"  # Set your test file path here

#==============================================================
TRAIN = True
# TRAIN = False
#False ##for evaluation
#==============================================================
MODEL_PATH = SAVE_PATH + "best_model600_0216_1"
#==============================================================
SEED=11
unlable_feat = []
USE_POS='--use_pos' 
USE_ENTITY_CLF='logits'     #Choices: none, logits, softmax, onehot

if __name__ == "__main__": 
    if TRAIN==True:  
      input_args_list = f"train --model_type {MODEL_TYPE} --label 'scierc_train' \
        --model_path  {BERT_PATH} --tokenizer_path  {BERT_PATH} \
        --train_path  {TRAIN_PATH} --valid_path {VALID_PATH} \
        --types_path {TYPES_PATH} --cache_path  {CACHE_PATH} \
        --size_embedding 25 --train_batch_size 1 \
        --use_pos true --pos_embedding 25 \
        --use_entity_clf {USE_ENTITY_CLF} \
        --eval_batch_size 1 --epochs 30 --lr 5e-5 --lr_warmup 0.1 \
        --weight_decay 0.01 --max_grad_norm 1.0 --prop_drop 0.1 \
        --neg_entity_count 100 --neg_relation_count 100 \
        --max_span_size 10 --rel_filter_threshold 0.4 --max_pairs 1000 \
        --sampling_processes 4 --sampling_limit 100  \
        --store_predictions True --store_examples True \
        --log_path  {LOG_PATH} --save_path {SAVE_PATH} \
        --max_seq_length 512 --config_path  {CONFIG_PATH}  --Names {names} \
        --gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph --seed {SEED} "

    else:
        input_args_list = f"eval --model_type {MODEL_TYPE} --label 'scierc_eval' \
          --model_path  {MODEL_PATH} --tokenizer_path {MODEL_PATH} \
          --dataset_path {TEST_PATH}  \
          --types_path  {TYPES_PATH}  --cache_path {CACHE_PATH} \
          --size_embedding 25 \
           {USE_POS} --pos_embedding 25 \
          --use_entity_clf {USE_ENTITY_CLF} \
          --eval_batch_size 1  --lr 5e-5 --lr_warmup 0.1 \
          --weight_decay 0.01 --max_grad_norm 1.0 --prop_drop 0.1 \
          --max_span_size 10 --rel_filter_threshold 0.4 --max_pairs 1000 \
          --sampling_processes 4 --sampling_limit 100 \
          --store_predictions True --store_examples True \
          --log_path {LOG_PATH} --save_path {SAVE_PATH} \
          --max_seq_length 512 --config_path {CONFIG_PATH} \
          --gradient_accumulation_steps 1  --wordpiece_aligned_dep_graph --seed {SEED}" #\
#          --no_overlapping"
        
    print("*** Commandline: ", input_args_list)

    input_args_list = input_args_list.split() 
    #print(input_args_list)
    r = Runner.Runner()
    r.run(input_args_list)

