export CUDA_VISIBLE_DEVICES=0

model_name="neulab/omnitab-large"
dataset_name="squall"
output_dir="output/squall_plus_tableqa0"
checkpoint=2850

python ./run.py \
  --task tableqa \
  --do_predict \
  --squall_plus True \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --predict_with_generate \
  --num_beams 5 \
  --input_noise 0
  
# --aug True 
# --max_predict_samples 500
