export GLUE_DIR=/home/ubuntu/sarcasm2020/tweet_data/fold1
export TASK_NAME=SST-2

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $GLUE_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 96 \
  --per_gpu_eval_batch_size 96 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --save_steps 47 \
  --overwrite_cache \
  --overwrite_output_dir \
  --output_dir bert_fine_tune_output
