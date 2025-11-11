import torch
import torch.nn as nn
from transformers import AutoModel

class SarcasmModel(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.2):
        super(SarcasmModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embed = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embed / sum_mask

    def forward(self, input_ids, attention_mask, is_pretraining=False):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if is_pretraining:
            hidden = self.mean_pool(output.last_hidden_state, attention_mask)
            return hidden
        else:
            hidden = output.last_hidden_state[:, 0, :]
            pooled_output = self.dropout(hidden)
            logits = self.classifier(pooled_output)
            return logits