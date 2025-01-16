# This script runs pyagent on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled
# - each query will be executed 5 times to do self-consistency

CUDA_VISIBLE_DEVICES=0 python run_agent_general.py \
    --model gpt-4o-mini \
    --provider openai --dataset tabfact \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent4 --cache_dir cache/gpt-3.5 \
