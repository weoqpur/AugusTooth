from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result