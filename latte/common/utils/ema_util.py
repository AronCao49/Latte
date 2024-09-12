import torch
import torch.nn as nn
import torchsparse.nn as spnn
from copy import deepcopy

def configure_model_norm(model: nn.Module, reset_dropout: bool) -> list:
    """
    Configure model for use with Tent, link: https://github.com/DequanWang/tent
    Args:
        model: input model that to be configured

    Returns:
        model: model with batch norm layers configured
    """
    model.train()
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    norm_list = []
    dropout_modules = []
    for name, m in model.named_modules():
        # check whether the module type belongs batch norm layers
        # print(type(m).__name__.lower())
        if 'norm' in type(m).__name__.lower():
            m.requires_grad_(True)
            norm_list.append(name)
            if 'batchnorm' in type(m).__name__.lower():
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # elif 'layernrom' in type(m).__name__.lower():
            #     m.reset_parameters()
        # reset dropout to 0.0 if required
        elif reset_dropout and 'dropout' in type(m).__name__.lower():
            m.p = 0.0
            dropout_modules.append(name)
        else:
            continue
    out = [model, [norm_list]]
    if reset_dropout:
        out[1].append([dropout_modules])
    return out


def configure_model_all(model: nn.Module, reset_dropout: bool) -> list:
    """
    Configure model for use with Tent, link: https://github.com/DequanWang/tent
    Args:
        model: input model that to be configured

    Returns:
        model: model with batch norm layers configured
    """
    model.train()
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    dropout_modules = []
    for name, m in model.named_modules():
        # check whether the module type belongs batch norm layers
        if 'norm' in type(m).__name__.lower():
            # enable grad and clear accumulated mean & var
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif reset_dropout and 'dropout' in type(m).__name__.lower():
            m.p = 0.0
            dropout_modules.append(name)
        else:
            m.requires_grad_(True)
    out = [model, [dropout_modules]]
    return out


def create_ema_model(model: nn.Module) -> list:
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    
    # Disable dropout layer
    dropout_modules = []
    for name, m in ema_model.named_modules():
        if 'dropout' in type(m).__name__.lower():
            m.p = 0.0
            dropout_modules.append(name)
    
    out = [ema_model, [dropout_modules]]
    
    return out


def update_ema_norm(
    ema_model: nn.Module, 
    model: nn.Module, 
    alpha_teacher: float, 
    iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for m, ema_m in zip(model.modules(), ema_model.modules()):
    # check whether the module type belongs batch norm layers
        if 'norm' in type(m).__name__.lower():
            # enable grad and clear accumulated mean & var
            for ema_param, param in zip(ema_m.parameters(), m.parameters()):
                #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        else:
            continue

    return ema_model


def ema_student_model(model, anchor, alpha_teacher=0.99):
    for nm, m in model.named_modules():
        if 'norm' in type(m).__name__.lower():
            for npp, p in m.named_parameters():
                with torch.no_grad():
                    p.data = anchor[f"{nm}.{npp}"] * (1 - alpha_teacher) + p * alpha_teacher


def reset_model(model, anchor):
    for nm, m in model.named_modules():
        if 'norm' in type(m).__name__.lower():
            for npp, p in m.named_parameters():
                with torch.no_grad():
                    p.data = anchor[f"{nm}.{npp}"]


def update_ema_all(
    ema_model: nn.Module, 
    model: nn.Module, 
    alpha_teacher: float, 
    iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model


def print_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad)