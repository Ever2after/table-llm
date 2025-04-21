import re
import numpy as np
import pandas as pd
import json
import logging
from utils.metric import check_match
from utils.col_processor import col_processor
import sqlite3

connection = sqlite3.connect(":memory:")

def evaluate_sql_result(pred_string: str, gold_string: str, target_delimiter=', '):
    correct_flag = check_match(pred_string, 'yes' if gold_string == '1' else 'no')
    return correct_flag

def execute_sql_on_table(pred_sql: str, table_info: dict):
    table_content = {
        "header": table_info['header'],
        "rows": table_info['rows']
    }

    table_content['header'] = col_processor(table_content['header'])

    table_content["rows"] = [[str(cell).strip().lower() if isinstance(cell, str) else cell for cell in row] for row in table_content["rows"]]

    table = pd.DataFrame(table_content["rows"], columns=table_content["header"])
    
    try:
        table.to_sql('my_table', connection, index=False, if_exists="replace")  # 데이터프레임을 "data"라는 테이블로 저장
        # SQL 실행
        result = pd.read_sql_query(pred_sql, connection)
        # 결과를 셀 데이터만 추출해서 리스트로 변환
        results = result.values.tolist()
        result_str = ', '.join([str(item).strip().lower() for row in results for item in row])
        if not result_str:
            result_str = "none"
        error = ""
    except Exception as e:
        # 오류 발생 시 반환
        logging.warning(f"SQL 실행 실패: {e}")
        result_str = "none"
        error = str(e)
    
    return result_str, error
    

def postprocess_sql_predictions(decoded_preds, eval_dataset, fuzzy=False):
    L = len(decoded_preds)
    results = []  # SQL 실행 결과 string
    errors = []
    for i, pred_sql in enumerate(decoded_preds):
        # 간단 로깅
        _id = eval_dataset['id'][i]
        logging.warning(f'{i}/{L}: {str(_id)}')
        logging.warning(f'Raw SQL: {pred_sql}')

        table = eval_dataset['table'][i]

        result_str, error = execute_sql_on_table(pred_sql, table)

        results.append(result_str)
        errors.append(error)
        print('\n---------------------\n')
    return results, errors


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=False, custom=False):
    
    def compute_metrics(eval_preds, meta=None):
        preds, labels = eval_preds

        # (1) preds가 tuple인 경우 정리
        if isinstance(preds, tuple):
            preds = preds[0]

        # (2) -100 (pad) → tokenizer.pad_token_id 치환
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
                if custom:
                    pred = pred.split("'''")[1]
                    pred = pred.split("'''")[0]
                    preds.append(pred.strip())
                    continue
                else:
                    pred = pred.split('Final Answer: ')[1]
                    # sql문 파싱 (```sql ... ``` -> ...)
                    pred = re.sub(r'```sql\n', '', pred)
                    pred = re.sub(r'```', '', pred)
                    pred = pred.replace('`', '')
                    pred = pred.strip()
                    preds.append(pred)
            except:
                try:
                    pred = pred.split('"code":')[1].strip()
                    pred = pred.split('"')[1]
                    pred = pred.split('"')[0]
                    preds.append(pred)
                except:
                    preds.append("")


        sql_results, errors = postprocess_sql_predictions(preds, eval_dataset, fuzzy)

        correct_flags = []
        for i, result_str in enumerate(sql_results):
            gold = eval_dataset["label"][i]
            if isinstance(gold, list):
                gold_string = ", ".join([g.strip().lower() for g in gold])
            else:
                gold_string = str(gold).strip().lower()

            # compare
            flag = evaluate_sql_result(result_str, gold_string, target_delimiter=', ')
            correct_flags.append(flag)

        acc = np.mean(correct_flags)

        # (5) CSV 저장 (stage)
        if stage:
            to_save = {
                'id': eval_dataset['id'],
                'question': ["Based on given table, check the following statement is true or false. Statement: " + statement for statement in eval_dataset['statement']],
                'answer': ['yes' if label == 1 else 'no' for label in eval_dataset['label']],
                'acc': [int(b) for b in correct_flags],
                'sql_pred': preds,         # 모델이 만든 SQL
                'sql_result': sql_results,         # SQL 실행 결과
                'error': errors,            # SQL 실행 에러
                'truncated': eval_dataset['truncated'],
                'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids']),
            }
            if meta:
                to_save['log_probs_sum'] = meta.get('log_probs_sum', [])
                to_save['log_probs_avg'] = meta.get('log_probs_mean', [])

            try:  
                df = pd.DataFrame(to_save)
                df.to_csv(f'./predict/tabfact/{stage}.csv', na_rep='', index=False)
                print('predictions saved! (csv) ', stage)
            except Exception as e:
                logging.warning(f"CSV 저장 실패: {e}")
                # json 형태로 저장
                with open(f'./predict/tabfact/{stage}.json', 'w') as f:
                    json.dump(to_save, f)
                    print('predictions saved! (json) ', stage)

        return {"acc": np.round(acc, 4)}

    return compute_metrics
