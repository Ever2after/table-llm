export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=STQA_wikisql
export WANDB_ENTITY=

model_name="neulab/omnitab-large"
run_name="wikisql_tableqa4"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa4"

python ./run.py \
  --do_train \
  --do_eval \
  --dataset_name ${dataset_name} \
  --split_id 4 \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 6 \
  --num_train_epochs 10 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --fp16 \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --eval_steps 100 \
  --save_steps 100 \
  --num_beams 5 \
  --generation_max_length 128 \
  --run_name ${run_name} \
  --task tableqa \
  --output_dir ${output_dir} \
  --save_total_limit 1 