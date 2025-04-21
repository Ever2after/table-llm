# This script runs pyagent on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled
# - each query will be executed 5 times to do self-consistency

CUDA_VISIBLE_DEVICES=0 python run_agent_general.py \
    --model google/gemma-2-2b-it \
    --provider huggingface --dataset tablebench --split test \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
    --log_dir output/tablebench_test_agent_gemma2_2b --cache_dir cache/tablebench_test_agent_gemma2_2b \

# CUDA_VISIBLE_DEVICES=0 python run_agent_general.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --provider vllm --dataset tabfact --split test \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
#     --log_dir output/tabfact_test_agent_llama --cache_dir cache/tabfact_test_agent_llama \


    # --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
