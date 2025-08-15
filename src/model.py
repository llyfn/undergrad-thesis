import torch.nn as nn
from transformers import AutoModel

class SarcasmModel(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.2):
        super(SarcasmModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, is_pretraining=False):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state[:, 0, :]

        if is_pretraining: return hidden
        else:
            pooled_output = self.dropout(hidden)
            logits = self.classifier(pooled_output)
            return logits