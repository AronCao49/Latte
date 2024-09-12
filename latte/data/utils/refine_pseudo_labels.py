import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from typing import List

from latte.models.losses import prob_2_entropy


def refine_pseudo_labels(probs, pseudo_label, ignore_label=-100) -> torch.Tensor:
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    if type(probs).__name__ == 'ndarray':
        probs = torch.tensor(probs)
    if type(pseudo_label).__name__ == 'ndarray':
        pseudo_label = torch.tensor(pseudo_label)
    for cls_idx in pseudo_label.unique():
        curr_idx = pseudo_label == cls_idx
        curr_idx = torch.nonzero(curr_idx).squeeze(1)
        thresh = probs[curr_idx].median()
        thresh = min(thresh, 0.9)
        ignore_idx = curr_idx[probs[curr_idx] < thresh]
        pseudo_label[ignore_idx] = ignore_label
    return pseudo_label


def refine_ety(ety, pseudo_label, conf_perc, ignore_label=-100) -> torch.Tensor:
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    all_mask = torch.ones_like(ety, dtype=torch.bool).cuda()
    if type(ety).__name__ == 'ndarray':
        ety = torch.tensor(ety)
    if type(pseudo_label).__name__ == 'ndarray':
        pseudo_label = torch.tensor(pseudo_label)
    for cls_idx in pseudo_label.unique():
        curr_idx = pseudo_label == cls_idx
        curr_idx = torch.nonzero(curr_idx).squeeze(1)
        thresh = torch.quantile(ety[curr_idx], conf_perc)
        class_invalid = curr_idx[ety[curr_idx] > thresh]
        all_mask[class_invalid] = False
    return all_mask


def generate_pslabel(cfg,
                     # 2D outputs
                     ori_prob_2d: torch.Tensor,
                     ori_pred_2d: torch.Tensor,
                     seg_logit_2d: torch.Tensor,
                     # 3D outputs
                     ori_prob_3d: torch.Tensor,
                     ori_pred_3d: torch.Tensor,
                     seg_logit_3d: torch.Tensor,
                     filter_type="PROTOTYPE",
                     # addition feat level input for prototype filtering
                     prototype_2d=None,
                     ori_feats_2d=None,
                     ema_feats_2d=None,
                     prototype_3d=None,
                     ori_feats_3d=None,
                     ema_feats_3d=None,
                     ) -> tuple:
    """
    Filtering function to generate reliable MM pseudo label
    """
    if filter_type == "HARD":
        ori_prob_2d, ori_pred_2d = ori_prob_2d.cpu().numpy(), ori_pred_2d.cpu().numpy()
        ori_prob_3d, ori_pred_3d = ori_prob_3d.cpu().numpy(), ori_pred_3d.cpu().numpy()
        ori_prob_2d = ori_prob_2d.max(1)
        ori_prob_3d = ori_prob_3d.max(1)
        # Option 1: confident 3D pred & less confident 2D pred
        mask_cfd_3d = np.logical_and(ori_prob_3d > cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_up, 
                                    ori_prob_2d < cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_low).astype(np.int64)
        # Option 2: confident 2D pred & less confident 3D pred
        mask_cfd_2d = np.logical_and(ori_prob_2d > cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_up, 
                                    ori_prob_3d < cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_low).astype(np.int64)
        # Option 3: both confident pred
        # mask_cfd_all = np.logical_and(ori_prob_2d > cfg.TTDA.MMCOTTA.tao_up, 
        #                               ori_prob_3d > cfg.TTDA.MMCOTTA.tao_up).astype(np.int64)
        # Option 4: both unreliable pred
        mask_cfd_non = np.logical_and(ori_prob_2d < cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_up, 
                                    ori_prob_3d < cfg.TTDA.MMCOTTA.PS_FILTER.HARD.tao_low).astype(np.int64)
        # summarize
        aug_pred_2d = seg_logit_2d.argmax(dim=1).cpu().numpy()
        aug_pred_3d = seg_logit_3d.argmax(dim=1).cpu().numpy()
        ps_label_2d = mask_cfd_3d * ori_pred_3d + \
                    mask_cfd_2d * ori_pred_2d + \
                    mask_cfd_non * (aug_pred_2d) + \
                    (1- mask_cfd_3d - mask_cfd_2d - mask_cfd_non) * \
                    ((seg_logit_2d + seg_logit_3d).argmax(dim=1).cpu().numpy())
        ps_label_3d = mask_cfd_3d * ori_pred_3d + \
                    mask_cfd_2d * ori_pred_2d + \
                    mask_cfd_non * (aug_pred_3d) + \
                    (1- mask_cfd_3d - mask_cfd_2d - mask_cfd_non) * \
                    ((seg_logit_2d + seg_logit_3d).argmax(dim=1).cpu().numpy())
        ps_label_2d = torch.from_numpy(ps_label_2d).cuda()
        ps_label_3d = torch.from_numpy(ps_label_3d).cuda()

        return ps_label_2d, ps_label_3d

    elif filter_type == "ADAPTIVE":
        ori_prob_2d, ori_pred_2d = ori_prob_2d.cpu().numpy(), ori_pred_2d.cpu().numpy()
        ori_prob_3d, ori_pred_3d = ori_prob_3d.cpu().numpy(), ori_pred_3d.cpu().numpy()
        ori_prob_2d = ori_prob_2d.max(1)
        ori_prob_3d = ori_prob_3d.max(1)
        ps_label_2d = np.ones_like(ori_pred_2d) * -100
        ps_label_3d = np.ones_like(ori_pred_3d) * -100
        # compute entropy-based weights
        rv_ety_2d = 1 / prob_2_entropy(F.softmax(seg_logit_2d, dim=1)).sum(1)
        rv_ety_3d = 1 / prob_2_entropy(F.softmax(seg_logit_3d, dim=1)).sum(1)
        weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
        weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
        # Intra-class ps-label refinement
        for preds, probs, aug_probs, ps_label in [(ori_pred_2d, ori_prob_2d, seg_logit_2d, ps_label_2d), 
                                                  (ori_pred_3d, ori_prob_3d, seg_logit_3d, ps_label_3d)]:
            for class_id in range(cfg.MODEL_2D.NUM_CLASSES):
                # select logit belong to category
                keep_index = preds == class_id
                if not np.any(keep_index):
                    continue
                # copy index for conf-based selection
                fuse_index = deepcopy(keep_index)
                augavg_index = deepcopy(keep_index)
                conf_index = deepcopy(keep_index)
                # conf-based selection
                probs_cat = probs[keep_index]
                adap_thre_up = np.percentile(probs_cat, cfg.TTDA.MMCOTTA.PS_FILTER.ADAPTIVE.perc_up)        # default 80%
                adap_thre_bt = np.percentile(probs_cat, cfg.TTDA.MMCOTTA.PS_FILTER.ADAPTIVE.perc_bot)       # default 20%
                augavg_index[augavg_index] = probs_cat <= adap_thre_bt
                fuse_index[fuse_index] = np.logical_and(probs_cat > adap_thre_bt, probs_cat <= adap_thre_up)
                conf_index[conf_index] = probs_cat > adap_thre_up
                # use aug-avg if not confident, and fuse if semi-confident
                # if only have one point, use aug-avg as ps-label
                ps_label[augavg_index] = aug_probs[augavg_index].argmax(dim=1).cpu().numpy()
                if np.any(fuse_index):
                    ps_label[fuse_index] = (seg_logit_2d[fuse_index] * weight_2d[fuse_index] + \
                                            seg_logit_3d[fuse_index] * weight_3d[fuse_index]).argmax(dim=1).cpu().numpy()
                if np.any(conf_index):
                    ps_label[conf_index] = preds[conf_index]
            
        # check whether all samples have been filtered
        assert np.all(ps_label_2d != -100)
        assert np.all(ps_label_3d != -100)
        ps_label_2d = torch.from_numpy(ps_label_2d).cuda()
        ps_label_3d = torch.from_numpy(ps_label_3d).cuda()

        return ps_label_2d, ps_label_3d
    
    elif filter_type == "PROTOTYPE":
        # normlize feat & prototype
        ori_feats_2d = ori_feats_2d / torch.linalg.norm(ori_feats_2d, dim=1, keepdim=True)
        ori_feats_3d = ori_feats_3d / torch.linalg.norm(ori_feats_3d, dim=1, keepdim=True)
        ema_feats_2d = ema_feats_2d / torch.linalg.norm(ema_feats_2d, dim=1, keepdim=True)
        ema_feats_3d = ema_feats_3d / torch.linalg.norm(ema_feats_3d, dim=1, keepdim=True)
        prototype_2d = prototype_2d / torch.linalg.norm(prototype_2d, dim=1, keepdim=True)
        prototype_3d = prototype_3d / torch.linalg.norm(prototype_3d, dim=1, keepdim=True)

        # compute distacne matrix
        ori_dist_2d = torch.exp(torch.matmul(ori_feats_2d, torch.t(prototype_2d)))
        ori_dist_3d = torch.exp(torch.matmul(ori_feats_3d, torch.t(prototype_3d)))
        ema_dist_2d = torch.exp(torch.matmul(ema_feats_2d, torch.t(prototype_2d)))
        ema_dist_3d = torch.exp(torch.matmul(ema_feats_3d, torch.t(prototype_3d)))

        # select prediction distance
        # original weight
        ori_weight_2d = torch.gather(ori_dist_2d, 1, ori_pred_2d.unsqueeze(-1))
        ori_weight_3d = torch.gather(ori_dist_3d, 1, ori_pred_3d.unsqueeze(-1))
        ori_weight_2d = ori_weight_2d / ori_dist_2d.sum(1, keepdim=True)
        ori_weight_3d = ori_weight_3d / ori_dist_3d.sum(1, keepdim=True)
        # aug-avg weight
        ema_pred_2d = seg_logit_2d.argmax(1)
        ema_pred_3d = seg_logit_2d.argmax(1)
        ema_weight_2d = torch.gather(ema_dist_2d, 1, ema_pred_2d.unsqueeze(-1))
        ema_weight_3d = torch.gather(ema_dist_3d, 1, ema_pred_3d.unsqueeze(-1))
        ema_weight_2d = ema_weight_2d / ema_dist_2d.sum(1, keepdim=True)
        ema_weight_3d = ema_weight_3d / ema_dist_3d.sum(1, keepdim=True)
        
        # intra-modal aggregation (iMPA)
        ps_prob_2d = (ori_weight_2d * ori_prob_2d + ema_weight_2d * seg_logit_2d) / (ori_weight_2d + ema_weight_2d)
        ps_prob_3d = (ori_weight_3d * ori_prob_3d + ema_weight_3d * seg_logit_3d) / (ori_weight_3d + ema_weight_3d)
        
        # class wise filtering
        ps_label_2d = ps_prob_2d.argmax(1)
        ps_label_3d = ps_prob_3d.argmax(1)
        
        # inter-modal aggregation (xMPF)
        weight_2d = torch.cat((ori_weight_2d, ema_weight_2d), dim=1).max(1, keepdim=True)[0]
        weight_3d = torch.cat((ori_weight_3d, ema_weight_3d), dim=1).max(1, keepdim=True)[0]
        # sum if pred is the same
        sum_idx_2d = ori_pred_2d == ema_pred_2d
        sum_idx_3d = ori_pred_3d == ema_pred_3d
        weight_2d[sum_idx_2d] = ori_weight_2d[sum_idx_2d] + ema_weight_2d[sum_idx_2d]
        weight_3d[sum_idx_3d] = ori_weight_3d[sum_idx_3d] + ema_weight_3d[sum_idx_3d] 
        ps_prob_xm = (weight_2d * ps_prob_2d + weight_3d * ps_prob_3d) / (weight_2d + weight_3d)

        # additional filtering
        mask_3d = F.softmax(ps_prob_3d, dim=1).max(1)[0] < cfg.TRAIN.COMAC.conf_thre
        mask_2d = F.softmax(ps_prob_2d, dim=1).max(1)[0] < cfg.TRAIN.COMAC.conf_thre
        mask_xm = torch.logical_and(mask_2d, mask_3d)
        # ps_prob_xm[mask_2d] = ps_prob_3d[mask_2d]
        # ps_prob_xm[mask_3d] = ps_prob_2d[mask_3d]

        ps_prob_xm = F.softmax(ps_prob_xm, dim=1)
        # Two option to filter out unreliable samples
        # unreliable_idx = ps_prob_xm.max(1)[0] < cfg.TTDA.MMCOTTA.PS_FILTER.PROTOTYPE.conf_thre
        # unreliable_idx = torch.logical_and(mask_2d, mask_3d)
        ps_label_xm = ps_prob_xm.argmax(1)
        label_xMPF = deepcopy(ps_label_xm)
        ps_label_xm[mask_xm] = -100

        return label_xMPF, ps_label_xm, F.softmax(ps_prob_2d, dim=1), F.softmax(ps_prob_3d, dim=1), mask_xm
    else:
        raise AssertionError("Current filter type is not defined: {}".format(filter_type))


def update_class_queue(
    queue: List[torch.Tensor],
    src_queue: List[torch.Tensor],
    feats: torch.Tensor,
    seg_logit: torch.Tensor,
    conf_thre=0.69,
    restore_prob=0.4,
    update_max_len=200,
    restore_max_len=200,
    ):
    # filter out unreliable samples based on prediction score
    conf_mask = seg_logit.max(1)[0] > conf_thre
    anchor_feat = []
    anchor_label = []
    # update class queue
    for i in range(len(queue)):
        # randomly sample reliable feats
        src_c_queue = src_queue[i]
        queue_c_len = queue[i].shape[0]
        mask_class = seg_logit.argmax(1) == i
        mask_class = torch.logical_and(mask_class, conf_mask)

        # stocastic restore source feats
        restore_idx = np.random.random()
        if restore_idx <= restore_prob:
            idx = torch.randperm(src_c_queue.shape[0])[:restore_max_len]
            queue[i] = queue[i][restore_max_len:, :]
            queue[i] = torch.cat([queue[i], src_c_queue[idx, :].detach()], dim=0)
            # queue[i][:(queue_c_len-update_max_len), :] = queue[i][update_max_len:, :].clone()
            # queue[i][-update_max_len:, :] = src_c_queue[idx, :].clone().detach()

        # check if any sample exists
        if not torch.any(mask_class):
            continue
        # feature normalization
        feat_c = feats[mask_class]
        feat_c = feat_c / torch.linalg.norm(feat_c, dim=1, keepdim=True)
        # update queue with maximum num of samples
        if feat_c.shape[0] > update_max_len:
            # Option 1: use the highest prob
            logit_c = torch.max(seg_logit, dim=1)[0][mask_class]
            _, sorted_idx = torch.sort(logit_c, dim=0, descending=True)
            idx = sorted_idx[:update_max_len]
            # Option 2: use the random idx
            # idx = torch.randperm(feat_c.shape[0])[:update_max_len]

            queue[i] = queue[i][update_max_len:, :]
            queue[i] = torch.cat([queue[i], feat_c[idx, :].clone().detach()], dim=0)
            # queue[i][:(queue_c_len-update_max_len), :] = queue[i][update_max_len:, :].clone()
            # queue[i][-update_max_len:, :] = feat_c[idx, :].clone().detach()
            anchor_feat.append(feat_c[idx, :])
            anchor_label.append(torch.ones(update_max_len) * i)
        else:
            feat_len = feat_c.shape[0]
            queue[i] = queue[i][feat_len:, :]
            queue[i] = torch.cat([queue[i], feat_c.clone().detach()], dim=0)
            # queue[i][:(queue_c_len-feat_len), :] = queue[i][feat_len:, :].clone()
            # queue[i][-feat_len:, :] = feat_c.clone().detach()
            anchor_feat.append(feat_c)
            anchor_label.append(torch.ones(feat_len) * i)

    return queue, torch.cat(anchor_feat, dim=0), torch.cat(anchor_label, dim=0)