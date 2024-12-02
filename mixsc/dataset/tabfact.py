import pandas as pd

from utils.data_loader import download_and_cache_dataset


class TabFactDataset:
    def __init__(self, cache_dir='data', split='test'):
        self.cache_dir = cache_dir
        self.split = split
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset_url = 'ibm/tab_fact'
        local_filename = f'tabfact_{self.split}.jsonl'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir, self.split)

    def get_item(self, index):
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset)} entries.")
        
        row = self.dataset.iloc[index]
        table = pd.DataFrame(row.table['rows'], columns=row.table['header'])
        question = f"Answer whether the following statement is true or false: {row['statement']}"
        answer = 'true' if row['label'] == 1 else 'false'

        return {
            "table": table,
            "question": question,
            "answer": answer
        }

    def __len__(self):
        return len(self.dataset)
