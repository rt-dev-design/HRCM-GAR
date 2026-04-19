# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ZimResNet18(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(
        self, 
        args
    ):
        super(ZimResNet18, self).__init__()
        
        assert args.backbone in ('resnet18'), "Zim only supports resnet18 for now"
        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            pretrained=args.use_pretrained_cnn
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.select_layer = [False, False, False, False, False]
        for layer_index in args.scale_selection_from_cnn:
            self.select_layer[layer_index] = True

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.select_layer[1]:
            output_list.append(x)
        x = self.layer2(x)
        if self.select_layer[2]:
            output_list.append(x)
        x = self.layer3(x)
        if self.select_layer[3]:
            output_list.append(x)
        x = self.layer4(x)
        if self.select_layer[4]:
            output_list.append(x)

        return output_list

# zim cnn
# multiscale information with corresponding position embeddings for a window
def build_zim_cnn(args):
    return ZimResNet18(args)
