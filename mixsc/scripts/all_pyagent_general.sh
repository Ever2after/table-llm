# This script runs pyagent on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled
# - each query will be executed 5 times to do self-consistency

export HF_HOME=/mnt/data/jusang/.cache/huggingface
export HF_TOKEN=

DATASET="tablebench"

# CUDA_VISIBLE_DEVICES=0 python run_agent_general.py \
#     --model google/gemma-2-27b-it \
#     --provider vllm --dataset "$DATASET" --split train \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
#     --log_dir output/${DATASET}_test_agent_gemma2_27b --cache_dir cache/${DATASET}_test_agent_gemma2_27b

CUDA_VISIBLE_DEVICES=0 python run_agent_general.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --provider vllm --dataset "$DATASET" --split train \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
    --log_dir output/${DATASET}_test_agent_qwen2.5_32b --cache_dir cache/${DATASET}_test_agent_qwen2.5_32b


    # --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
