import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model

from dataset.tabmwp import TabMWPDataset
from dataset.wikitq import WikiTQDataset
from dataset.tabfact import TabFactDataset
from dataset.general import GeneralDataset

from run_helper import get_cot_prompt, query

def main(
        model:Optional[str] = "gpt-3.5-turbo", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo", # long model of the agent (only used for long prompt)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq or tabfact
        self_consistency:int = 10, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/wtq_cot", # directory to store the logs
):
    
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset and cot prompt ####
    if dataset == "wtq":
        data = WikiTQDataset(cache_dir='data', split='test')
    elif dataset == "tabfact":
        data = TabFactDataset(cache_dir='data', split='test')
    elif dataset == "tabmwp":
        data = TabMWPDataset(cache_dir='data', split='test')
    else:
        data = GeneralDataset(cache_dir='data', split='test')
    
    cot_prompt = get_cot_prompt(dataset)

    #### load the model ####
    if model:
        model = Model(model, provider=provider)
    if long_model:
        long_model = Model(long_model, provider=provider)
    
    total = len(data)
    global_i = 0
    pbar = tqdm(total=total)

    for idx in range(total):
        d = data.get_item(idx)
        question = d["question"]
        answer = d["answer"]
        table = d["table"].to_markdown()

        prompt = cot_prompt.replace("[TABLE]", table)\
            .replace("[QUESTION]", question)\
            .strip()

        text, response = query(model, long_model, prompt, temperature, self_consistency)
        
        res = {
            "idx": global_i,
            "answer": [answer],
            "text": text,
            "table": table,
            "question": question,
        }

        with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
            json.dump(res, f)
            f.write("\n")

        global_i += 1
        pbar.update(1)

if __name__ == "__main__":
    Fire(main)