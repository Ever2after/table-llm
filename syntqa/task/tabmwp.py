import json
import datasets

class TabmwpDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="TabMWP dataset containing table-based math word problems.",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),  # 리스트나 null 가능성 고려
                    "answer": datasets.Value("string"),
                    "unit": datasets.Value("string"),
                    "table_title": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    # table_for_pd는 JSON 형태를 그대로 문자열로 저장
                    "table_for_pd": datasets.Value("string"),
                    "row_num": datasets.Value("int32"),
                    "column_num": datasets.Value("int32"),
                    "solution": datasets.Value("string"),
                    "ques_type": datasets.Value("string"),
                    "ans_type": datasets.Value("string"),
                    "grade": datasets.Value("int32"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://example.com",  # 필요시 수정
            citation="...citation...",        # 필요시 수정
        )

    def _split_generators(self, dl_manager):
        """
        하나의 JSON 파일에 train/test/validation 등 여러 스플릿 정보가 함께 들어있다고 가정하고,
        'split' 필드에 따라 데이터를 나누어 반환합니다.
        """
        # 로컬 파일 경로 예시(또는 URL로 제공 가능)
        train_data_path = "data/tabmwp/tabmwp_train.json"
        test_data_path = "data/tabmwp/tabmwp_test.json"
        
        # URL인 경우 (예시)
        # data_url = "https://raw.githubusercontent.com/.../tabmwp.json"
        # data_path = dl_manager.download_and_extract(data_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_data_path, "target_split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "target_split": "test"},
            ),
            # validation이 있다면 추가
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={"filepath": data_path, "target_split": "validation"},
            # ),
        ]

    def _generate_examples(self, filepath, target_split):
        """
        filepath로부터 JSON을 읽어오고, 각 문제에 대해 (idx, data_dict)를 yield합니다.
        target_split(예: "train", "test")에 해당하는 항목만 필터링합니다.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)  # { "8": {...}, "9": {...}, ... }

        idx = 0
        for ex_id, ex_data in data.items():
            # split 필드로 분기하여 해당 스플릿만 골라냄
            if ex_data.get("split") == target_split:
                yield idx, {
                    "id": ex_id,
                    "question": ex_data.get("question", ""),
                    # choices가 None이면 빈 리스트로 처리
                    "choices": ex_data["choices"] if ex_data["choices"] else [],
                    "answer": ex_data.get("answer", ""),
                    "unit": ex_data.get("unit", ""),
                    "table_title": ex_data.get("table_title", ""),
                    "table": ex_data.get("table", ""),
                    # JSON 구조인 table_for_pd는 문자열로 저장
                    "table_for_pd": json.dumps(ex_data.get("table_for_pd", {})),
                    "row_num": ex_data.get("row_num", 0),
                    "column_num": ex_data.get("column_num", 0),
                    "solution": ex_data.get("solution", ""),
                    "ques_type": ex_data.get("ques_type", ""),
                    "ans_type": ex_data.get("ans_type", ""),
                    "grade": ex_data.get("grade", 0),
                    "split": ex_data.get("split", "")
                }
                idx += 1

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/tabmwp.py")
    sample = dataset["train"][0]
    print(sample)