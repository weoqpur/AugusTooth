import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.upsample(output, size=(h, w), mode="bilinear")

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
