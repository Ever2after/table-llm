export CUDA_VISIBLE_DEVICES=0


export HF_HOME=/mnt/data/jusang/.cache/huggingface
export HF_TOKEN=

# model_name="t5-large"
# model_name="google/gemma-2-2b-it"
model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="meta-llama/Llama-3.2-3B-Instruct"
dataset_name="wikitq"
output_dir="output/wikitq_text_to_sql1"
checkpoint=4600

python ./run_vllm.py \
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
  --max_predict_samples 10 \

  # --input_noise 9
  

  # --squall_downsize 2

# --save_note nor 

# --max_predict_samples 100
# --aug True 
# --max_predict_samples 500  

  # --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
