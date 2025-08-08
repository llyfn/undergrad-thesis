import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Union
from transformers import AutoTokenizer

class SarcasmDataset(Dataset):
    def __init__(self, data_source: Union[str, pd.DataFrame], model_name, max_length: int):
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source
        self.data = self.data.dropna(subset=['comment']).reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.comments = self.data['comment'].tolist()
        self.parent_comments = self.data['parent_comment'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self) -> int:
        return len(self.comments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comment = str(self.comments[idx])
        parent_comment = str(self.parent_comments[idx])
        label = int(self.labels[idx])

        input_text = f"{parent_comment} [SEP] {comment}" if parent_comment else comment

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': input_text
        }