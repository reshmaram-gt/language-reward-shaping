import torch
import torch.nn as nn

class NCELoss(nn.Module):
    def __init__(self, temp=1):
        super(NCELoss, self).__init__()
        self.temp = temp
        
    def forward(self, pos_scores, neg_scores):
        
        pos_scores = pos_scores / self.temp
        neg_scores = neg_scores / self.temp
        max_scores = torch.max(torch.concatenate((pos_scores, neg_scores), axis=1), axis=1)[0].unsqueeze(1)
        nce = torch.exp(pos_scores - max_scores) / (torch.exp(pos_scores - max_scores) + torch.exp(neg_scores - max_scores))
        return -torch.sum(torch.log(nce), axis=0)