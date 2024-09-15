import torch
import torch.nn as nn

from latte.models.segformer_utils.backbone import mit_b0, mit_b1, mit_b2
from latte.models.segformer_utils.seg_head import SegFormerHead, resize

class SegFormer(nn.Module):
    """Simple re-factored version of SegFormer without MMLab libs
    
    Official repo: https://github.com/NVlabs/SegFormer/
    
    """
    
    def __init__(
        self, 
        backbone: str = "mit_b2", 
        decoder_cfg: dict = {}, 
    ):
        super(SegFormer, self).__init__()
        if backbone.lower() == "mit_b0":
            self.backbone = mit_b0()
        elif backbone.lower() == "mit_b1":
            self.backbone = mit_b1()
        elif backbone.lower() == "mit_b2":
            self.backbone = mit_b2()
        else:
            raise KeyError(
                "The required backbone type {} is not implemented!".format(backbone))
            
        self.decode_head = SegFormerHead(**decoder_cfg)
        
    def forward(self, x):
        ori_shape = x.shape
        
        # encode-decode
        feats = self.backbone(x)
        x = self.decode_head(feats)
        
        # interpolation
        x = resize(x, ori_shape[2:], mode='bilinear', align_corners=False)
        
        return x


if __name__ == "__main__":
    net = SegFormer()
    net = net.cuda()
    
    x = torch.randn(3, 3, 400, 225)
    x = x.cuda()
    print(x.shape)
    
    out = net(x)
    print(out.shape)