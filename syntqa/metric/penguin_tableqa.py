import numpy as np
import pandas as pd
import logging
from utils.metric import check_match
import json

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None, custom=False): 
    def compute_metrics(eval_preds, meta=None):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100s used for padding as we can't decode them
        # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds_no_minus100 = []
        for seq in preds:
            # seq는 길이가 다른 토큰 시퀀스
            # seq 내에서 -100을 pad_token_id로 바꾸기
            seq_no_minus100 = [tok if tok != -100 else tokenizer.pad_token_id for tok in seq]
            preds_no_minus100.append(seq_no_minus100)
        decoded_preds = tokenizer.batch_decode(preds_no_minus100, skip_special_tokens=True)

        # '\n\nFinal Answer: 1' -> '1' 로 파싱 (예외처리도)
        preds = []
        for pred in decoded_preds:
            try:
                pred = pred.split('Final Answer: ')[1]
                preds.append(pred)
            except:
                preds.append("")
            
        
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # prepare the prediction format for the evaluator

        preds, labels = postprocess_text(preds, eval_dataset['answer'])

        correct_flag = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            correct_flag.append(check_match(pred, label))
            # print(f'pred: {pred}, label: {label}, correct: {correct_flag[-1]}')

        if stage:
            to_save = {
                       'id': eval_dataset['id'],
                       'question': eval_dataset['question'],
                       'answer': labels,
                       'acc': [int(b) for b in correct_flag],
                       'predictions': preds,
                       'truncated': eval_dataset['truncated'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean']

            try:
                df = pd.DataFrame(to_save)
                df.to_csv(f'./predict/penguin/{stage}.csv', na_rep='',index=False)
                print('predictions saved! ', stage)
            except Exception as e:
                logging.warning(f"CSV 저장 실패: {e}")
                # json 형태로 저장
                with open(f'./predict/penguin/{stage}.json', 'w') as f:
                    json.dump(to_save, f)
                    print('predictions saved! (json) ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics


