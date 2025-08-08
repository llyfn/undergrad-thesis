import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_hidden, teacher_hidden, labels):
        student_hidden = F.normalize(student_hidden, dim=-1)
        teacher_hidden = F.normalize(teacher_hidden, dim=-1)

        similarity_matrix = torch.matmul(student_hidden, teacher_hidden.T) / self.temperature

        batch_size = student_hidden.size(0)
        positive_mask = torch.eye(batch_size, device=student_hidden.device)

        logits = similarity_matrix - positive_mask * 1e9
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        positive_log_prob = log_prob[positive_mask.bool()].view(batch_size, -1)
        loss = -positive_log_prob.mean()
        return loss