# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python3 run_cot_general.py \
    --model gpt-3.5-turbo --long_model gpt-3.5-turbo \
    --provider openai --dataset general \
    --self_consistency 1 --temperature 0.8 \
    --log_dir output/general_dp
