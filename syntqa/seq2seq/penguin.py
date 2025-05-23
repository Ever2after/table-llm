import sys
sys.path.append('./')
from copy import deepcopy
from utils.processor import get_default_processor, get_custom_processor
from utils.col_processor import col_processor
import json
from io import StringIO
import pandas as pd

def preprocess_function(examples,
                               tokenizer,
                               max_source_length,
                               max_target_length,
                               ignore_pad_token_for_loss,
                               padding,
                               input_noise=None,
                               custom=False):

    if custom:
        TABLE_PROCESSOR = get_custom_processor(
            max_cell_length=10000,
            max_input_length=10000,
            target_delimiter=', '
        )
    else:
        TABLE_PROCESSOR = get_default_processor(
            max_cell_length=10000,
            max_input_length=10000,
            target_delimiter=', '
        )

    inputs = []          # 인코더 입력용 텍스트 (질문+테이블)
    outputs = []         # 디코더 라벨(예: solution 또는 sql 등)
    input_truncated = [] # 테이블이 트렁케이트되었는지 여부 추적

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        table = pd.read_csv(StringIO(examples["table"][i]))
    
        table_content = {
            "header": table.columns.tolist(),
            "rows": table.values.tolist()
        }

        table_content['header'] = col_processor(table_content['header'])

        table_content_copy = deepcopy(table_content)
        
        if custom:
            input_source = TABLE_PROCESSOR.process_input(
            table_content_copy,
            question,
            [],
            False,
            True,
            False).strip().lower()
        else:        
            input_source = TABLE_PROCESSOR.process_input(
                table_content_copy, 
                question, 
                []
            ).strip().lower() + '''\nLet's think step by step, and then give the SQL query to solve the problem. Ensure the final answer format is only "Final Answer: {sql query}" form, no other form. The table name is 'my_table'.''' #  E.g. "Final Answer: select col1 from my_table where col2 = 'value'".
        
        # (4) 테이블이 트렁케이트되었는지 여부
        input_truncated.append(False)

        # (5) 최종 모델 입력용 문자열
        inputs.append(input_source)

        # (6) 모델이 출력해야 할 라벨(텍스트) 결정
        output = TABLE_PROCESSOR.process_output(answer).lower()
        outputs.append(output)


    # (5) 모델 입력용 토큰화
    model_inputs = tokenizer(
        text=inputs, 
        max_length=max_source_length, 
        padding=padding, 
        truncation=True
    )
    
    # (6) 라벨(정답) 토큰화
    labels = tokenizer(
        text=outputs,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    
    # (7) padding 토큰 -100 처리(모델 학습 시 PAD 토큰 무시)
    if padding == "max_length" and ignore_pad_token_for_loss:
        pad_token_id = tokenizer.pad_token_id
        for label_ids in labels["input_ids"]:
            for j in range(len(label_ids)):
                if label_ids[j] == pad_token_id:
                    label_ids[j] = -100
    
    # (8) 최종 결과
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["truncated"] = [int(x) for x in input_truncated]
    
    return model_inputs


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset_dict = load_dataset("./task/penguin.py")
    train_dataset = dataset_dict["test"].select(range(100))  # 일부분만 샘플

    # (2) T5Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")

    # (3) map 함수를 통해 전처리
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": 1024,
            "max_target_length": 512,
            "ignore_pad_token_for_loss": True,
            "padding": False # 또는 False/True
        },
        batched=True,  # examples가 batch 단위로 들어옴
    )

    # (4) 전처리 결과 확인
    i = 0
    print("Decoded input_ids :", tokenizer.decode(train_dataset["input_ids"][i]))
    print("Decoded labels    :", tokenizer.decode(train_dataset["labels"][i]))
    print("Truncated flag    :", train_dataset["truncated"][i])
