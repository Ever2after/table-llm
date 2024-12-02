import os
import json
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model

from dataset.tabmwp import TabMWPDataset
from dataset.wikitq import WikiTQDataset
from dataset.tabfact import TabFactDataset
from dataset.general import GeneralDataset

def main(
        model:Optional[str] = "gpt-3.5-turbo", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo", # long model of the agent (only used for long prompt)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq, tabfact
        self_consistency:int = 1, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/tabfact_agent", # directory to store the logs
):
    
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset ####
    if dataset == "wtq":
        data = WikiTQDataset(cache_dir='data', split='test')
    elif dataset == "tabfact":
        data = TabFactDataset(cache_dir='data', split='test')
    elif dataset == "tabmwp":
        data = TabMWPDataset(cache_dir='data', split='test')
    else:
        data = GeneralDataset(cache_dir='data', split='test')

    #### load the model ####
    if model:
        model = Model(model, provider=provider)
    if long_model:
        long_model = Model(long_model, provider=provider)
    
    
    #### prepare the iterator ####
    total = len(data)
    global_i = 0
    pbar = tqdm(total=total)
    
    #### start the loop ####
    for idx in range(total):
        d = data.get_item(idx)
        question = d["question"]
        answer = d["answer"]
        table = d["table"]
            
        log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
        # create the file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        texts = []
            
        for _ in range(self_consistency):  
            # create the table agent
            agent = TableAgent(
                table=table,
                prompt_type=dataset,
                model=model,
                long_model=long_model,
                temperature=temperature,
                log_dir=log_path,
                use_full_table=True,
            )

            text, response = agent.run(question=question, title="")
            texts.append(text)


        res = {
            "idx": global_i,
            "answer": [answer],
            "text": texts if self_consistency > 1 else texts[0],
            "table": table.to_markdown(),
            "question": question,
        }

        with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
            json.dump(res, f)
            f.write("\n")
                
        global_i += 1
        pbar.update(1)

if __name__ == "__main__":
    Fire(main)