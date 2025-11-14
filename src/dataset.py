import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

FIGLANG_REDDIT_TRAIN = './data/figlang_reddit/sarcasm_detection_shared_task_reddit_training.jsonl'
FIGLANG_REDDIT_TEST = './data/figlang_reddit/sarcasm_detection_shared_task_reddit_testing.jsonl'
FIGLANG_TWITTER_TRAIN = './data/figlang_twitter/sarcasm_detection_shared_task_twitter_training.jsonl'
FIGLANG_TWITTER_TEST = './data/figlang_twitter/sarcasm_detection_shared_task_twitter_testing.jsonl'

class SarcasmDataset(Dataset):
    def __init__(self, data, model_name, max_length: int):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['text']
        label = item['label']

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

def load_dataset(dataset, model_name, val_ratio=0.1, max_length=256, random_state=42):
    if dataset == 'figlang-reddit':
        train, test = FIGLANG_REDDIT_TRAIN, FIGLANG_REDDIT_TEST
    else:
        train, test = FIGLANG_TWITTER_TRAIN, FIGLANG_TWITTER_TEST

    with open(train, 'r', encoding='utf-8') as f:
        train_val_data = [json.loads(line) for line in f]
    with open(test, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=random_state, stratify=[d['label'] for d in train_val_data])

    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)
    test_data = preprocess_data(test_data)

    train_dataset = SarcasmDataset(train_data, model_name, max_length)
    val_dataset = SarcasmDataset(val_data, model_name, max_length)
    test_dataset = SarcasmDataset(test_data, model_name, max_length)

    return train_dataset, val_dataset, test_dataset

def preprocess_data(data):
    processed_data = []
    label_map = {"SARCASM": 1, "NOT_SARCASM": 0}
    for item in data:
        context = " [SEP] ".join(item['context'])
        comment = item['response']
        input_text = f"{context} [SEP] {comment}" if context else comment
        processed_data.append({
            'text': input_text,
            'label': label_map[item['label']]
        })
    return processed_data
