export CUDA_VISIBLE_DEVICES=0

# model_name="t5-large"
model_name="Qwen/Qwen2.5-Coder-14B-Instruct"
dataset_name="tabmwp"
output_dir="output/tabmwp_text_to_sql1"
checkpoint=4600

python ./run2.py \
  --task text_to_sql \
  --do_predict \
  --squall_plus True \
  --predict_split test \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --postproc_fuzzy_string True \
  --max_source_length 4096 \
  --max_target_length 2048 \
  --val_max_target_length 2048 \
  --per_device_eval_batch_size 1 \
  --dataset_name ${dataset_name} \
  --split_id 5 \
  --predict_with_generate \
  --num_beams 1 \
  --max_predict_samples 1000

  # --input_noise 9
  

  # --squall_downsize 2

# --save_note nor 

# --max_predict_samples 100
# --aug True 
# --max_predict_samples 500  

  # --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
