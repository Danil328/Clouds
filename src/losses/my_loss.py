import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse
from metrics import dice_coef


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, target):
        return 1.0 - dice_coef(inputs, target, smooth=self.smooth)


class SoftLogDiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, target):
        return -torch.log(dice_coef(inputs, target, self.smooth))


class BCEDiceLossWithLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftLogDiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, target) + self.dice(inputs, target)


class BCEDiceLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.dice_loss = SoftDiceLoss()
        self.bce = nn.BCELoss()
        self.bce_weight = weights['bce']
        self.dice_weight = weights['dice']

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, target) * self.bce_weight + self.dice_loss(inputs, target) * self.dice_weight


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, add_weight=False, pos_weight=2.0, neg_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.add_weight = add_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, inputs, targets):
        if self.add_weight and self.logits==False:
            weights = targets.clone()
            weights[(targets == 0.0) & (inputs >= 0.5)] = self.pos_weight
            weights[(targets == 0.0) & (inputs < 0.5)] = self.neg_weight
            weights[(targets == 1.0) & (inputs >= 0.5)] = self.neg_weight
            weights[(targets == 1.0) & (inputs < 0.5)] = self.neg_weight
        else:
            weights = None

        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, weight=weights)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False, weight=weights)

        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class FocalDiceLossWithLog(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.dice_loss = SoftLogDiceLoss()
        self.bce = FocalLoss(logits=False, add_weight=False)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, target) * self.bce_weight + self.dice_loss(inputs, target) * self.dice_weight


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.dice_loss = SoftDiceLoss()
        self.bce = FocalLoss(logits=False, add_weight=False)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, target) * self.focal_weight + self.dice_loss(inputs, target) * self.dice_weight


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, add_weight=False, pos_weight=2.0, neg_weight=1.0, focal_weight=1.0, tversky_weight=1.0):
        super().__init__()
        self.dice_loss = TverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = FocalLoss(logits=False, add_weight=add_weight, pos_weight=pos_weight, neg_weight=neg_weight)
        self.focal_weight = focal_weight
        self.dice_weight = tversky_weight

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, target) * self.focal_weight + self.dice_loss(inputs, target) * self.tversky_weight



class FocalTverskyLossKernel(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLossKernel, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=2.0):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class TverskyLoss(nn.Module):

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        batch_size = input.size(0)
        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1)

        # compute the actual dice score
        intersection = torch.sum(input * target, dim=1)
        fps = torch.sum(input * (1 - target), dim=1)
        fns = torch.sum((1 - input) * target, dim=1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_score = (numerator + self.eps) / (denominator + self.eps)
        tversky_loss = (1 - tversky_score) ** self.gamma
        return tversky_loss.mean()


"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
#PyTorch

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

#=====
#Multi-class Lovasz loss
#=====

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

#PyTorch
class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)
        return Lovasz


# # PyTorch
# CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
# class ComboLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(ComboLoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         # True Positives, False Positives & False Negatives
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#
#         inputs = torch.clamp(inputs, e, 1.0 - e)
#         out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
#         weighted_ce = out.mean(-1)
#         combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
#
#         return combo


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


if __name__ == '__main__':
    predict = torch.tensor(((0.01, 0.03, 0.02, 0.02),
                            (0.05, 0.12, 0.09, 0.07),
                            (0.89, 0.85, 0.88, 0.91),
                            (0.99, 0.97, 0.95, 0.97)), dtype=torch.float)

    target = torch.tensor(((0, 0, 0, 0),
                           (0, 0, 0, 0),
                           (1, 1, 1, 1),
                           (1, 1, 1, 1)), dtype=torch.float)

    inter = torch.sum(predict * target)

    dice = 2 * inter / (torch.sum(predict) + torch.sum(target))

    dice_coef(predict.unsqueeze(0), target.unsqueeze(0))
