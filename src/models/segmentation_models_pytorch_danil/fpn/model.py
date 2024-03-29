from .decoder import FPNDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder
from torch import nn

class FPN(EncoderDecoder):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        dropout: spatial dropout rate in range (0, 1).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        final_upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            classes=1,
            dropout=0.2,
            activation='sigmoid',
            final_upsampling=4,
            cls_out=0,
            hypercolumn=False,
            supervision=False,
            attentionGate=False
    ):
        hypercolumn = True if hypercolumn=='true' else False
        supervision = True if supervision=='true' else False
        attentionGate = True if attentionGate=='true' else False

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        if cls_out > 0:
            cls = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.1),
                nn.Linear(56, 24),
                nn.BatchNorm1d(24),
                nn.ELU(),
                nn.Linear(24, cls_out),
                nn.Sigmoid() if activation=='sigmoid' else nn.Softmax()
            )
        else:
            cls = None

        decoder = FPNDecoder(
            encoder_channels=encoder.out_shapes,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            final_channels=classes,
            dropout=dropout,
            final_upsampling=final_upsampling,
            hypercolumn=hypercolumn,
            supervision=supervision,
            attentionGate=attentionGate,
        )

        super().__init__(encoder, decoder, activation, cls)

        self.name = 'fpn-{}'.format(encoder_name)
