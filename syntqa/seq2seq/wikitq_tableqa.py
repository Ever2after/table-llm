import sys
sys.path.append('./')
from copy import deepcopy
from utils.processor import get_default_processor
import json

def preprocess_function(
    examples,
    tokenizer,
    max_source_length,
    max_target_length,
    ignore_pad_token_for_loss,
    padding,
    input_noise=None,
    custom=False
):
    # 원하는 파라미터에 맞춰 Processor 생성 (예시)
    TABLE_PROCESSOR = get_default_processor(
        max_cell_length=1000, 
        max_input_length=2048, 
        target_delimiter=', '
    )
    
    input_sources = []
    output_targets = []
    input_truncated = []
    
    # batch 단위로 들어온 examples에서 하나씩 꺼내 처리
    for i in range(len(examples['question'])):
        # (1) 문제, 정답, 테이블 정보 가져오기
        question = examples['question'][i] 
        answer = examples['answers'][i] 
        table = examples['table'][i]    
        
        table_content = {
            "header": table['header'],
            "rows":  table['rows']
        }
        
        # deepcopy로 복제
        table_content_copy = deepcopy(table_content)
        
        # (2) 스플릿에 따라 테이블 필터링/정답 노출 등 결정
        current_split = examples['split'][i]  # 예: 'train' or 'test'
        
        if current_split == "train":
            # 훈련 시에는 answer를 table processor에 넘겨 
            # 불필요한 행을 필터링하거나, 정답 정보를 활용할 수 있음
            input_source = TABLE_PROCESSOR.process_input(
                table_content_copy, 
                question, 
                answer
            ).lower()
        else:
            # 평가/테스트 시에는 정답 정보를 숨긴 상태로 입력 구성
            input_source = TABLE_PROCESSOR.process_input(
                table_content_copy, 
                question, 
                []
            ).lower() + '''\nLet's think step by step, and then give the final answer. Ensure the final answer format is only "Final Answer: AnswerName1, AnswerName2..." form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.'''
        
        input_sources.append(input_source)

        # (3) truncate 여부 판단(선택적)
        # 실제 row가 input_source 안에 모두 들어있는지 간단하게 체크하는 로직
        if len(table_content_copy["rows"]) > 0:
            last_cell = str(table_content_copy["rows"][-1][-1]).lower().strip()[:15]
            n_row = len(table_content_copy["rows"])
            truncated = (
                (f'row {n_row}' not in input_source) or 
                (last_cell not in input_source.split('|')[-1].strip())
            )
        else:
            truncated = False
        input_truncated.append(truncated)

        # (4) 타겟(정답) 토큰화
        # process_output를 통해 answer를 문자열로 만든 뒤 토크나이저 적용
        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

    # (5) 모델 입력용 토큰화
    model_inputs = tokenizer(
        text=input_sources, 
        max_length=max_source_length, 
        padding=padding, 
        truncation=True
    )
    
    # (6) 라벨(정답) 토큰화
    labels = tokenizer(
        text=output_targets,
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

    dataset_dict = load_dataset("task/wikitq.py")  # or {"train":..., "test":...}

    # 샘플로 일부만 사용해보기
    train_dataset = dataset_dict["test"].select(range(1000))

    # (2) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")

    # (3) 전처리 함수 매핑
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": 1024,
            "max_target_length": 512,
            "ignore_pad_token_for_loss": True,
            "padding": False,
        },
        batched=True,   # 여러 샘플을 한번에 처리(주의: TABLE_PROCESSOR가 batch도 대응 가능해야 함)
    )

    # (4) 결과 확인
    i = 0
    print("Decoded input_ids:", tokenizer.decode(train_dataset['input_ids'][i]))
    print("Decoded labels   :", tokenizer.decode(train_dataset['labels'][i]))
    print("Truncated flag   :", train_dataset['truncated'][i])
