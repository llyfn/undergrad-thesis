import torch
import torch.nn as nn
from transformers import AutoModel

class SarcasmModel(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.2):
        super(SarcasmModel, self).__init__()
        self.student = AutoModel.from_pretrained(model_name)
        self.teacher = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.student.config.hidden_size, num_labels)
        self.teacher.eval()
        for param in self.teacher.parameters(): param.requires_grad = False

    def forward(self, input_ids, attention_mask, is_pretraining=False):
        student_output = self.student(input_ids=input_ids, attention_mask=attention_mask)
        student_hidden = student_output.last_hidden_state[:, 0, :]
        if is_pretraining:
            with torch.no_grad():
                teacher_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_hidden = teacher_output.last_hidden_state[:, 0, :]
            return student_hidden, teacher_hidden
        else:
            pooled_output = self.dropout(student_hidden)
            logits = self.classifier(pooled_output)
            return logits