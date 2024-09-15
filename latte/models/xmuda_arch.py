import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latte.models.segformer import SegFormer
from latte.models.segformer_utils.seg_head import resize
from latte.models.spvcnn import SPVCNN
from torch.autograd import Function


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def batch_segment(xm_feats, batch_masks):
    # segment xm_feats in to list based on the batch_masks: [TENSOR_B1,...,TENSOR_Bm]
    qkv_embed_list = []
    start_flag = 0
    max_len = 0
    for mask in batch_masks:
        embed = xm_feats[start_flag : (start_flag+mask)]
        qkv_embed_list.append(embed)
        max_len = embed.shape[0] if embed.shape[0] > max_len else max_len
        # update start_flag
        start_flag += mask
    return qkv_embed_list, max_len

def batch_assemble(attn_feats, batch_masks):
    # re-assemble the features and reshape the batch in to first dimension
    re_feat_list = []
    for i in range(len(batch_masks)):
        feat = attn_feats[:batch_masks[i],i,:]
        re_feat_list.append(feat)
    return torch.cat(re_feat_list, dim=0)

def adaptive_cat(qkv_embed_list, max_len):
    # pre-assign the output tensor of the shape [L, N, E]
    # out_tensor = torch.zeros(max_len, len(qkv_embed_list), qkv_embed_list[0].shape[1]).cuda()
    # attn_mask = torch.BoolTensor(len(qkv_embed_list), max_len).cuda()

    for i in range(len(qkv_embed_list)):
        # adaptive concatenation
        qkv_embed = qkv_embed_list[i]
        point_num = qkv_embed.shape[0]
        if point_num < max_len:
            qkv_embed = torch.cat((qkv_embed, torch.zeros(max_len-point_num, qkv_embed.shape[1]).cuda()), dim=0)
        if i == 0:
            out_tensor = qkv_embed.unsqueeze(1)
        else:
            out_tensor = torch.cat((out_tensor,qkv_embed.unsqueeze(1)), dim=1)
        # generate attn_mask
        mask = torch.sum(out_tensor[:,i,:], dim=-1) != 0
        if i == 0:
            attn_mask = mask.unsqueeze(0)
        else:
            attn_mask = torch.cat((attn_mask,mask.unsqueeze(0)), dim=0)

    attn_mask = torch.matmul(attn_mask.unsqueeze(2).float(), attn_mask.unsqueeze(1).float())
    attn_mask = attn_mask == 0
    return out_tensor, attn_mask

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs,
                 da_method=None,
                 output_all=False,
                 ):
        super(Net2DSeg, self).__init__()
        self.test_cfg = backbone_2d_kwargs.pop('test_cfg')
        
        # 2D image network
        if backbone_2d == 'SegFormer':
            pretrained_path = backbone_2d_kwargs.pop("pretrained_path")
            self.net_2d = SegFormer(**backbone_2d_kwargs)
            feat_channels = self.net_2d.decode_head.embedding_dim
            # segmentation head
            self.linear = nn.Linear(feat_channels, num_classes)
            if pretrained_path:
                print("Loading 2D pretrained from {}".format(pretrained_path))
                state_dict = torch.load(pretrained_path)['state_dict']
                miss_keys, unexpected_keys = self.net_2d.load_state_dict(state_dict, strict=False)
                print("Missing Keys: {}".format(miss_keys))
                print("Unexpected Keys: {}".format(unexpected_keys))
            else:
                print("Random init SegFormer")
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.output_all = output_all
        self.num_classes = num_classes
        self.feat_channels = feat_channels

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)
        
        self.da_method = da_method
        if da_method == "MCD":
            self.linear3 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch, reverse=False):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        x = self.net_2d(img)

        # 2D-3D feature lifting
        if self.output_all:
            x_all = x.clone().permute(0, 2, 3, 1)
            pred_all = self.linear(x_all)

        img_feats = []
        for i in range(x.shape[0]):
            img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)

        # linear
        preds = {'feats': img_feats}

        if self.output_all:
            preds['seg_logit_all'] = pred_all

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats)

        if self.da_method == "MCD":
            if reverse:
                img_feats = ReverseLayerF.apply(img_feats, 1)
            preds['seg_logit3'] = self.linear3(img_feats)

        if self.da_method == "MMD":
            preds['feats_all'] = x_all

        x = self.linear(img_feats)
        preds['seg_logit'] = x

        return preds
    
    # Inference code from mmseg/SegFormer
    def slide_inference(
        self, img: torch.Tensor, 
        img_meta: dict, 
        rescale: bool, 
        requires_grad: bool, 
        return_feats: bool = False
        ):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        if return_feats:
            feats = img.new_zeros((batch_size, self.feat_channels, h_img, w_img))
        else:
            feats = None
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_feats = self.net_2d(crop_img).permute(0, 2, 3, 1)
                crop_seg_logit = self.linear(crop_feats).permute(0, 3, 1, 2)
                preds = preds + F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                if return_feats:
                    feats = feats + F.pad(crop_feats.permute(0, 3, 1, 2),
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] = count_mat[:, :, y1:y2, x1:x2] + 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)
        if return_feats:
            feats = feats / count_mat
            if rescale:
                feats = resize(
                    feats,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=False,
                    warning=False)
        return preds, feats

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        
        feats = self.net_2d(img).permute(0, 2, 3, 1)
        seg_logit = self.linear(feats).permute(0, 3, 1, 2)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)

        return seg_logit

    def inference(self, data_dict, img_meta=None, rescale=False, requires_grad=False, return_all=False, return_feats=False):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        # ori_shape = img_meta[0]['ori_shape']
        # assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit, feats = self.slide_inference(data_dict['img'], img_meta, rescale, requires_grad, return_feats=return_feats)
        else:
            seg_logit = self.whole_inference(data_dict['img'], img_meta, rescale)

        img2pc_logit = []
        img2pc_feats = []
        img_indices = data_dict['img_indices']
        for i in range(seg_logit.shape[0]):
            img2pc_logit.append(seg_logit.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            if return_feats:
                img2pc_feats.append(feats.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img2pc_logit = torch.cat(img2pc_logit, 0)
        img2pc_feats = torch.cat(img2pc_feats, 0) if return_feats else img2pc_feats
        out = {'seg_logit': img2pc_logit, 'feats': img2pc_feats}
        
        # return all logits if needed
        if return_all:
            out.update({'all_logits_2d': seg_logit})
        
        return out


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 da_method=None,
                 pretrained=False
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        self.backbone_3d = backbone_3d
        if backbone_3d == 'SPVCNN':
            self.net_3d = SPVCNN(**backbone_3d_kwargs)
            if pretrained:
                state_dict = torch.load("latte/models/pretrained/cr05/init")['model']
                print("Loaded state from xmuda/models/pretrained/cr05/init")
                self.net_3d = load_state(self.net_3d, state_dict, strict=False)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        if "Base" not in backbone_3d:
            # segmentation head
            self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

            # 2nd segmentation head
            self.dual_head = dual_head
            if dual_head:
                self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)

            # da method
            self.da_method = da_method
            if da_method == "MCD":
                self.linear3 = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch, reverse=False):
        feats = self.net_3d(data_batch['lidar'])
        x = self.linear(feats)

        preds = {
            'feats': feats,
            'seg_logit': x,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats)

        if self.da_method == "MCD":
            if reverse:
                feats = ReverseLayerF.apply(feats, 1)
            preds['seg_logit3'] = self.linear3(feats)

        return preds


class ModalAttn(nn.Module):
    def __init__(self,
                 num_classes,
                 inplanes=16, 
                 num_heads=1,
                 inner_dim=32,
                 dropout=0.5,
                 mode="attn",
                 ):
        super(ModalAttn, self).__init__()
        self.pc_linear = nn.Sequential(nn.Linear(inplanes, inner_dim),
                                       nn.ReLU(),
                                       nn.Linear(inner_dim, 64),
                                       nn.ReLU()
                                      )
        self.mode = mode
        self.num_heads = num_heads
        if self.mode == "attn":
            self.inner_dim = inner_dim
            self.to_qkv = nn.Linear(128, 3*inner_dim)
            self.attn = torch.nn.MultiheadAttention(embed_dim=inner_dim,
                                                    num_heads=num_heads)
            self.fc_cls = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(inner_dim, num_classes))
        elif self.mode == "cat":
            self.cat_linear = nn.Sequential(nn.Dropout(p=dropout),
                                            nn.Linear(128, num_classes))

    def forward(self, img_feats, pc_feats, batch_masks=None):
        # print(img_feats.shape)
        # print(mask_indices.shape)
        pc_feats = self.pc_linear(pc_feats)
        xm_feats = torch.cat((pc_feats, img_feats), dim=1)

        if self.mode == "attn":
            attn_feats = self.to_qkv(xm_feats)
            # batch segment in to list of tensors
            qkv_embed_list, max_len = batch_segment(attn_feats, batch_masks)
            # adaptive concatenation and output attn_mask
            qkv_embed, attn_mask = adaptive_cat(qkv_embed_list, max_len)
            print("Shape of qkv_embed: {}".format(qkv_embed.shape))
            print("Shape of attn_mask: {}".format(attn_mask.shape))
            # qkv_embed = qkv_embed.unsqueeze(1)
            q, k, v = qkv_embed[:,:,0:self.inner_dim],\
                      qkv_embed[:,:,self.inner_dim:2*self.inner_dim],\
                      qkv_embed[:,:,2*self.inner_dim:]
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
            attn_feats, attn_weights = self.attn(q, k, v, need_weights=True, attn_mask=attn_mask)
            # assemble the prediction into a joint tensor, attn_feats of shape [L, N, E]
            attn_feats = batch_assemble(attn_feats, batch_masks)
            xm_preds = self.fc_cls(attn_feats)
        else:
            xm_preds= self.cat_linear(xm_feats)

        return {
                "xm_feats": xm_feats,
                "xm_preds": xm_preds
                }


# TODO: Complete downsampling for predictions
# class Downsample_SCN(nn.Module):
#     def __init__(self,
#                  inplanes=64,
#                  )

class Domain_CLF(nn.Module):
    def __init__(self,
                 inplanes=128,
                 dropout=0.0,
                 ):
        super(Domain_CLF, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(p=dropout),
                                nn.Linear(inplanes, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 2)
                                )

    def forward(self, x, alpha, batch_masks):
        x = ReverseLayerF.apply(x, alpha)
        feats, max_len = batch_segment(x, batch_masks)
        for i in range(len(feats)):
                feats[i] = torch.mean(feats[i], 0, True)
        feats = torch.cat(feats, dim=0)

        x = self.fc(feats)
        return x

# Adv classifier from https://github.com/thuml/CDAN/blob/c904cd3f2fe092bb5175be1c5b6e42fa2036dece/pytorch/network.py
class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature=1280, hidden_size=500):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)

    return out_dict, img_indices


def test_Net3DSeg(backbone='SCN'):
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))

    if backbone == 'SCN':
        feats = torch.rand(num_coords, in_channels)
        feats = feats.cuda()

        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SCN',
                          backbone_3d_kwargs={'in_channels': in_channels})

        net_3d.cuda()
        out_dict = net_3d({
            'x': [coords, feats],
        })
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)

    elif "SPVCNN" in backbone:
        from torchsparse import SparseTensor
        feats = torch.cat((coords, torch.rand(num_coords, in_channels)), dim=1)
        lidar = SparseTensor(feats, coords).cuda()
        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SPVCNN',
                          backbone_3d_kwargs={'cr': 0.5}).cuda()
        # Load pretrained test
        # import os
        # print(os.getcwd())
        # state_dict = torch.load("init")["model"]
        # net_3d = load_state(net_3d, state_dict)
        out_dict = net_3d({"lidar":lidar})
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)

    elif backbone == "SalsaNext":
        range_img = torch.rand(1, 5, 64, 2048).cuda()
        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SalsaNext',
                          backbone_3d_kwargs={'in_channels': in_channels}).cuda()
        out_dict = net_3d({"proj_in": range_img})
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)
    
    return out_dict


def load_state(net, state_dict, strict=True):
	if strict:
		net.load_state_dict(state_dict=state_dict)
	else:
		# customized partially load function
		net_state_keys = list(net.state_dict().keys())
		for name, param in state_dict.items():
			name_m = name if "module." not in name else name[7:]
			if name_m in net.state_dict().keys():
				dst_param_shape = net.state_dict()[name_m].shape
				if param.shape == dst_param_shape:
					net.state_dict()[name_m].copy_(param.view(dst_param_shape))
					net_state_keys.remove(name_m)
		# indicating missed keys
		if net_state_keys:
			print(">> Failed to load: {}".format(net_state_keys))
			return net
	return net


if __name__ == '__main__':
    # Test lines for xmuda
    # img_dict, img_indices = test_Net2DSeg()
    from torchsparse import SparseTensor
    pc_dict = test_Net3DSeg("SPVCNN")

    
    # Test lines for attention
    

    # print("Shape of CDAN outputs: {}".format(output.shape))




