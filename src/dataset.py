import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class SarcasmDataset(Dataset):
    def __init__(self, data_source, model_name, max_length: int):
        self.data = data_source
        self.data = self.data.dropna(subset=['comment']).reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.comments = self.data['comment'].tolist()
        self.parent_comments = self.data['parent_comment'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self) -> int:
        return len(self.comments)

    def __getitem__(self, idx):
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
            'label': torch.tensor(label, dtype=torch.long),
            'text': input_text
        }

def load_dataset(data_source, model_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, max_length=128, random_state=42):
    data = pd.read_csv(data_source)
    train_val_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42, stratify=data['label'])
    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio/(train_ratio + val_ratio), random_state=random_state, stratify=train_val_data['label'])

    train_dataset = SarcasmDataset(train_data, model_name, max_length)
    val_dataset = SarcasmDataset(val_data, model_name, max_length)
    test_dataset = SarcasmDataset(test_data, model_name, max_length)

    return train_dataset, val_dataset, test_dataset