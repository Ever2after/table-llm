export CUDA_VISIBLE_DEVICES=0

# model_name="neulab/omnitab-large"
model_name="Qwen/Qwen2.5-Coder-14B-Instruct"
dataset_name="tabmwp"
output_dir="output/tabmwp_tableqa1"
checkpoint=4050

python ./run2.py \
  --task tableqa \
  --do_predict \
  --squall_plus True \
  --predict_split test \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --max_source_length 4096 \
  --max_target_length 2048 \
  --val_max_target_length 2048 \
  --per_device_eval_batch_size 1 \
  --dataset_name ${dataset_name} \
  --split_id 2 \
  --predict_with_generate \
  --num_beams 1 \
  --max_predict_samples 1000

# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \

# --aug True 