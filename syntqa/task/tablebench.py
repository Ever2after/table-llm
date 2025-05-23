import json
import datasets

class TableBenchDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Tablebench",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "qtype": datasets.Value("string"),
                    "qsubtype": datasets.Value("string"),
                    "table": {
                        "columns": datasets.Sequence(datasets.Value("string")),
                        "data": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    },
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "chart_type": datasets.Value("string"),
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
        # train_data_path = "data/tablebench/tablebench_train.jsonl"
        test_data_path = "data/tablebench/tablebench_train.jsonl"
        
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
        """
        filepath로부터 JSON을 읽어오고, 각 문제에 대해 (idx, data_dict)를 yield합니다.
        target_split(예: "train", "test")에 해당하는 항목만 필터링합니다.
        """
        qsubtypes = { 
            'Aggregation',
            'ArithmeticCalculation',
            'Comparison',
            'CorrelationAnalysis',
            'Counting',
            'Domain-Specific',
            'ImpactAnalysis',
            'MatchBased',
            'Multi-hop FactChecking',
            'Multi-hop NumericalReasoing',
            'Ranking',
            'StatisticalAnalysis',
            'Time-basedCalculation',
            'TrendForecasting'
        }
        with open(filepath, "r", encoding="utf-8") as f:
            # jsonl 파일을 한 줄씩 읽어옴
            for idx, line in enumerate(f):
                data = json.loads(line)
                if data['qsubtype'] not in qsubtypes:
                    continue
                data['split'] = target_split
                yield idx, data

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/tablebench.py")
    sample = dataset["test"][0]
    print(sample)