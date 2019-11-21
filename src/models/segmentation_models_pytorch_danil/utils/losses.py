import torch
import torch.nn as nn
from . import functions as F
import torch.nn.functional as FF


class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


def criterion_mask(logit, truth, weight=None):
    if weight is None: weight=[1,1,1,1]
    weight = torch.FloatTensor([1]+weight).to(truth.device).view(1,-1)

    batch_size,num_class,H,W = logit.shape

    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)
    # return F.cross_entropy(logit, truth, reduction='mean')

    log_probability = -FF.log_softmax(logit,-1)
    probability = FF.softmax(logit,-1)

    onehot = torch.zeros(batch_size*H*W,num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1,1),value=1) #F.one_hot(truth,5).float()

    loss = log_probability*onehot

    probability = probability.view(batch_size,H*W,5)
    truth  = truth.view(batch_size,H*W,1)
    weight = weight.view(1,1,5)

    alpha  = 2
    focal  = torch.gather(probability, dim=-1, index=truth.view(batch_size,H*W,1))
    focal  = (1-focal)**alpha
    focal_sum = focal.sum(dim=[1,2],keepdim=True)
    #focal_sum = focal.sum().view(1,1,1)
    weight = weight*focal/focal_sum.detach() *H*W
    weight = weight.view(-1,5)


    loss = loss*weight
    loss = loss.mean()
    return loss