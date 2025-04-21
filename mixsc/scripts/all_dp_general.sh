# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
    --model google/gemma-2-2b-it \
    --provider huggingface --dataset tablebench --split test \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
    --log_dir output/tablebench_test_dp_gemma2_2b --cache_dir cache/tablebench_test_dp_gemma2_2b

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --provider huggingface --dataset tabfact --split test \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 1 --temperature 0.8 \
#     --log_dir output/tabfact_test_dp_deepseek --cache_dir cache/tabfact_test_dp_deepseek

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --provider huggingface --dataset tabmwp --split test \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 1 --temperature 0.8 \
#     --log_dir output/tabmwp_test_dp_deepseek --cache_dir cache/tabmwp_test_dp_deepseek

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model google/gemma-3-4b-it \
#     --provider huggingface --dataset wtq= --split test \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 1 --temperature 0.8 \
#     --log_dir output/wtq_test_dp_gemma --cache_dir cache/wtq_test_dp_gemma

    # --model Qwen/Qwen2.5-Coder-14B-Instruct \
    # --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \