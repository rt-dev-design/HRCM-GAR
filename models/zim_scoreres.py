import torch
import torch.nn.functional as F
from torch import nn, Tensor

class ScorerBase(nn.Module):

    def __init__(self):
        super(ScorerBase, self).__init__()

    def forward(self):
        raise NotImplementedError
    
    def softmax_weighted_sum(self, features, scores):
        # features: B x N x D, scores: B x N x 1
        attn = F.softmax(scores, dim=1)
        return torch.sum(attn * features, dim=1)

    def softmax_element_weighting(self, features, scores):
        # features: B x N x D, scores: B x N x 1
        attn = F.softmax(scores, dim=1)
        return attn * features
    
class LearnedDotProductScorer(ScorerBase):

    def __init__(self, vector_length):
        super(LearnedDotProductScorer, self).__init__()
        self.scorer = nn.Linear(vector_length, 1, bias=False)
    
    def forward(self, x):
        # x: B x N x D -> scores
        return self.scorer(x)


class MlpScorer(ScorerBase):

    def __init__(self, input_dim, scorer_hidden_dim=64, dropout=0.1):
        super(MlpScorer, self).__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: B x N x D -> scores
        return self.scorer(x)