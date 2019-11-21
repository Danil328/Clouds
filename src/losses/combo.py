import torch
from torch import nn

from losses.my_loss import TverskyLoss
from models.segmentation_models_pytorch_danil.utils.losses import criterion_mask
from .dice import DiceLoss
from .focal import FocalLoss2d, FocalLoss
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss, LovaszLossSigmoid


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, channel_weights=None, channel_losses=None, activation="sigmoid"):
        super().__init__()
        if channel_weights is None:
            channel_weights = [1, 1, 1, 1]
        self.weights = weights
        self.activation = activation

        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.focal_frog = criterion_mask
        self.focal_git = FocalLoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.tversky = TverskyLoss(alpha=0.5, beta=0.5, gamma=2.0)

        self.mapping = {
            'bce': self.bce,
            'ce': self.ce,
            "focal_frog": self.focal_frog,
            "focal_git": self.focal_git,
            'dice': self.dice,
            'focal': self.focal,
            'jaccard': self.jaccard,
            'lovasz': self.lovasz,
            'lovasz_sigmoid': self.lovasz_sigmoid,
            'tversky': self.tversky
        }

        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid', 'tversky', 'bce', 'ce'}
        self.per_channel = {'dice', 'jaccard', 'lovasz_sigmoid', 'tversky'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        if self.activation == 'sigmoid':
            sigmoid_input = torch.sigmoid(outputs)
        elif self.activation == 'softmax':
            sigmoid_input = torch.softmax(outputs, dim=1)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...]
                        )
            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss.clamp(min=1e-5)


class ComboSuperVisionLoss(ComboLoss):
    def __init__(self, weights, per_image=False, channel_weights=(1.0, 1.0, 1.0, 1.0), channel_losses=None, sv_weight=0.15, activation="sigmoid"):
        channel_weights = channel_weights if activation=='sigmoid' else (1.0, 1.0, 1.0, 1.0, 1.0)
        super().__init__(weights, per_image, channel_weights, channel_losses, activation)
        self.sv_weight = sv_weight

    def forward(self, *input):
        outputs, targets = input
        mask_loss = super().forward(outputs, targets)
        # supervision_loss = F.binary_cross_entropy(sv_outputs, sv_targets)
        # return self.sv_weight * supervision_loss + (1 - self.sv_weight) * mask_loss
        return mask_loss
