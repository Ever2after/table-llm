# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
    --model gpt-4o-mini --long_model gpt-4o-mini \
    --provider openai --dataset tabmwp \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_dp_general --cache_dir cache/gpt-4o-mini
