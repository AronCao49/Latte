import torch
from typing import List

def pc2img_logit(
    pc_logits: torch.Tensor,
    pc2img_idx_ls: List[torch.Tensor],
    return_mask: bool = False
    ) -> torch.Tensor:
    """Function to map full FOV pc logits to image's
    Arugments:
        1. pc_logits: N_pc x N_classes point-wise logits
        2. pc2img_idx_ls: [N_per_scan] bool map
    Return:
        torch.Tensor: pc logits withing image FOV
    """
    idx_start = 0
    mapped_pc_logits = []
    pc2img_mask = []
    shape_inv = False
    # reshape label
    if pc_logits.dim() == 1:
        pc_logits, shape_inv = pc_logits.reshape(-1,1), True
    for i, idx in enumerate(pc2img_idx_ls):
        curr_logits = pc_logits[idx_start:(idx_start + idx.shape[0]),:]
        mapped_pc_logits.append(curr_logits[idx, :])
        idx_start += idx.shape[0]
        pc2img_mask.append(idx)
    
    # Concate
    mapped_pc_logits = torch.cat(mapped_pc_logits, dim=0)
    mapped_pc_logits = mapped_pc_logits.reshape(-1) \
        if shape_inv else mapped_pc_logits
    
    if return_mask:
        return [torch.cat(pc2img_mask, dim=0), mapped_pc_logits]
    else:
        return mapped_pc_logits