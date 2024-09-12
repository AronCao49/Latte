import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


class BerhuLoss(nn.Module):
    """ Inverse Huber Loss """
    def __init__(self, ignore_index = 1):
        super(BerhuLoss, self).__init__()
        self.ignore_index = ignore_index
        self.l1 = torch.nn.L1Loss(reduction = 'none')

    def forward(self, prediction, ground_truth, imagemask=None):
        if imagemask is not None:
            mask = (ground_truth != self.ignore_index) & imagemask.to(torch.bool)
        else:
            mask = (ground_truth != self.ignore_index)
        difference = self.l1(torch.masked_select(prediction, mask), torch.masked_select(ground_truth, mask))
        with torch.no_grad():
            c = 0.2*torch.max(difference)
            mask = (difference <= c)

        lin = torch.masked_select(difference, mask)
        num_lin = lin.numel()

        non_lin = torch.masked_select(difference, ~mask)
        num_non_lin = non_lin.numel()

        total_loss_lin = torch.sum(lin)
        total_loss_non_lin = torch.sum((torch.pow(non_lin, 2) + (c**2))/(2*c))

        return (total_loss_lin + total_loss_non_lin)/(num_lin + num_non_lin)

def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
    # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
    L2_distance = ((total0-total1)**2).sum(2) # For feature MMD, use only a single sum(2)
    # L2_distance = ((total0-total1)**2).sum(2).sum(2) # For corr MMD, use two sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

# Lovasz Softmax
def isnan(x):
    return x != x

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    if probas.dim() == 4:
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)
    

# focal loss
def focal_loss(preds: torch.Tensor, labels: torch.Tensor, 
               alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
    """
    Focal loss used in RatinaNet.

    Args:
        preds (Tensor): A float tensor of shape [N, K].
        labels (Tensor): a float tensor of label [N, ].
        alpha (float): Weighting factor in range (0, 1) to 
                        balance positive and negative examples.
        gamma (float): Exponent of modulating factor to balance easy vs hard samples.
        reduction (str): reduction method.
    Return: 
        Loss tensor
    """
    # reshape labels and one-hot encoding
    labels = labels.reshape(-1, 1)
    target = torch.nn.functional.one_hot(labels.long())

    # use the official focal loss function
    loss = torchvision.ops.focal_loss(preds, target, 
                                      alpha=alpha, gamma=gamma, reduction=reduction)
    
    # return
    return loss

class SupConLoss(nn.Module):
    """
    Modified from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, 
                labels_anchor,
                anchor_feature, 
                contrast_feature, 
                labels_contrast):

        # class-wise contrastive loss
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()          # N_a x N_q
        # tile mask
        anchor_mask = labels_anchor.reshape(-1,1).repeat(1, labels_contrast.shape[0])
        contrast_mask = labels_contrast.reshape(-1,1).repeat(1, labels_anchor.shape[0])
        mask = torch.eq(anchor_mask, contrast_mask.T).float().cuda()

        # compute log_prob
        exp_logits = (torch.exp(logits) + 1e-5) * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)      # N_a / N_a

        # loss
        loss = -mean_log_prob_pos.mean()

        return loss