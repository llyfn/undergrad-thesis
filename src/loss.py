import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, hidden, labels):
        hidden = F.normalize(hidden, dim=-1)
        similarity_matrix = torch.matmul(hidden, hidden.T) / self.temperature

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        positive_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity_matrix)
        pos_sum = (exp_sim * positive_mask).sum(1) / (positive_mask.sum(1) + 1e-9)  # positives 평균 sim
        total_sum = exp_sim.sum(1)

        loss = -torch.log(pos_sum / total_sum).mean()
        return loss