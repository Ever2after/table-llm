import json
import datasets

class PenguinDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Penguin",
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "options": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "answer": datasets.Value("string"),
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
        test_data_path = "data/penguin/penguin_test.json"
        
        # URL인 경우 (예시)
        # data_url = "https://raw.githubusercontent.com/.../tabmwp.json"
        # data_path = dl_manager.download_and_extract(data_url)

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     gen_kwargs={"filepath": train_data_path, "target_split": "train"},
            # ),
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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = 0
        for ex_id, ex_data in data.items():
            yield idx, {
                "id": idx,
                "question": ex_data["Question"],
                "options": ex_data["Options"],
                "table": ex_data["Table"],
                "target": ex_data["target"],
                "answer": ex_data["answer_extract"],
                "split": target_split,
            }
            idx += 1


if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/penguin.py")
    sample = dataset["test"][0]
    print(sample)