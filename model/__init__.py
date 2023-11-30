import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet, resnet18, resnet34, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from config.config import cfg
from utils.weight_add_noise import weight_add_noise
from utils.weight_init import weight_init

from .unet3plus import UNet3Plus

resnets = ["resnet18", "resnet34", "resnet50", "resnet101"]
resnet_cfg = {
    "return_nodes": {
        "relu": "layer0",
        "layer1": "layer1",
        "layer2": "layer2",
        "layer3": "layer3",
        "layer4": "layer4",
    },
    "resnet18": {
        "fe_channels": [64, 64, 128, 256, 512],
        "channels": [32, 64, 128, 256, 512],
    },
    "resnet50": {
        "fe_channels": [64, 256, 512, 1024, 2048],
        "channels": [64, 128, 256, 512, 1024],
    },
}


class U3PResNetEncoder(nn.Module):
    """
    ResNet encoder wrapper
    """

    def __init__(self, backbone="resnet18", weights=None) -> None:
        super().__init__()

        self.resnet: ResNet = globals()[backbone](weights=weights)
        cfg = resnet_cfg["resnet18"] if backbone in ["resnet18", "resnet34"] else resnet_cfg["resnet50"]
        if weights == None:
            self.resnet.apply(weight_init)
        self.backbone = create_feature_extractor(self.resnet, return_nodes=resnet_cfg["return_nodes"])

        # print(self.resnet)
        # input = torch.randn(1, 3, 320, 320)
        # out = self.backbone(input)

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg["fe_channels"], cfg["channels"])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())
        self.channels = [3] + cfg["channels"]

    def forward(self, x):
        ori_backbon = copy.deepcopy(self.backbone)
        if cfg.model.fig == "A":
            if self.training:
                torch.cuda.manual_seed_all(cfg.model.noise_seed)
                torch.manual_seed(cfg.model.noise_seed)
                np.random.seed(cfg.model.noise_seed)
                random.seed(cfg.model.noise_seed)

                self.backbone.apply(weight_add_noise)

            torch.cuda.manual_seed_all(cfg.train.seed)
            torch.manual_seed(cfg.train.seed)
            np.random.seed(cfg.train.seed)
            random.seed(cfg.train.seed)

        elif cfg.model.fig == "C":
            if self.training:
                torch.cuda.manual_seed_all(cfg.model.noise_seed)
                torch.manual_seed(cfg.model.noise_seed)
                np.random.seed(cfg.model.noise_seed)
                random.seed(cfg.model.noise_seed)
            else:
                torch.cuda.manual_seed_all(cfg.train.seed)
                torch.manual_seed(cfg.train.seed)
                np.random.seed(cfg.train.seed)
                random.seed(cfg.train.seed)

            self.backbone.apply(weight_add_noise)

        out = self.backbone(x)
        for ii, compress in enumerate(self.compress_convs):
            out[f"layer{ii}"] = compress(out[f"layer{ii}"])
        out = [v for _, v in out.items()]

        del self.backbone
        self.backbone = ori_backbon
        return out


def build_unet3plus(
    num_classes,
    encoder="default",
    skip_ch=64,
    aux_losses=2,
    use_cgm=False,
    weights=None,
    dropout=0.3,
) -> UNet3Plus:
    if encoder == "default":
        encoder = None
        aux_losses = 4
        dropout = 0.0
        transpose_final = False
        fast_up = False
    elif encoder in resnets:
        encoder = U3PResNetEncoder(backbone=encoder, weights=weights)
        transpose_final = True
        fast_up = True
    else:
        raise ValueError(f"Unsupported backbone : {encoder}")
    model = UNet3Plus(
        num_classes,
        skip_ch,
        aux_losses,
        encoder,
        use_cgm=use_cgm,
        dropout=dropout,
        transpose_final=transpose_final,
        fast_up=fast_up,
    )
    return model
