import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-9

    def forward(self, hidden, labels):
        hidden = F.normalize(hidden, dim=-1)

        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        logits_mask = torch.ones_like(labels_matrix, dtype=torch.bool).fill_diagonal_(False)
        positive_mask = labels_matrix & logits_mask

        sim = torch.matmul(hidden, hidden.T) / self.temperature
        sim_no_diag = sim.masked_fill(torch.eye(hidden.shape[0], dtype=torch.bool, device=hidden.device), -self.eps)
        log_prob = sim_no_diag - torch.logsumexp(sim_no_diag, dim=1, keepdim=True)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + self.eps)
        loss = -mean_log_prob_pos.mean()

        return loss