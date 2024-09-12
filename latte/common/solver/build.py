"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR


def build_optimizer(optim_cfg, model, weight_decay=False, model_type="2D"):
    name = optim_cfg.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        param_base_layer = []
        name_base_layer = []
        param_new_layer = []
        name_new_layer = []
        # if cfg.MODEL_3D.TYPE == "SPVCNN" and cfg.MODEL_3D.SPVCNN.pretrained and model_type == "3D":
        #     for name_param, param in model.named_parameters():
        #         if "linear" in name_param:
        #             param_new_layer.append(param)
        #             name_new_layer.append(name_param)
        #         else:
        #             name_base_layer.append(name_param)
        #             param_base_layer.append(param)
        #     print(">>> Reduce lr_base for pretrained param: {}".format(str(name_base_layer[0:5]) + " ... " + str(name_base_layer[-5:])))
        #     return getattr(torch.optim, name)(
        #         [
        #             {'params':param_base_layer, 'lr':optim_cfg.BASE_LR / 10},
        #             {'params':param_new_layer}
        #         ],
        #         lr=optim_cfg.BASE_LR,
        #         weight_decay=optim_cfg.WEIGHT_DECAY,
        #         **optim_cfg.get(name, dict()),
        #     )
        # else:
        if name != "AdamW" or optim_cfg.get(name, dict())['param_keys'] == ():
            optim_dict = optim_cfg.get(name, dict())
            optim_dict.pop('mult_types', None) 
            optim_dict.pop('param_keys', None) 
            optim_dict.pop('mults', None) 
            return getattr(torch.optim, name)(
                model.parameters(),
                lr=optim_cfg.BASE_LR,
                weight_decay=optim_cfg.WEIGHT_DECAY,
                **optim_dict,
            )
        else:
            optim_dict, param_groups = param_spec_optimizer(model, optim_cfg)
            return getattr(torch.optim, name)(
                param_groups, 
                lr=optim_cfg.BASE_LR,
                weight_decay=optim_cfg.WEIGHT_DECAY,
                **optim_dict
            )
    else:
        raise ValueError('Unsupported type of optimizer.')

def param_spec_optimizer(model: torch.nn.Module, optim_cfg):
    param_groups =[]
    all_spec_names = []
    base_lr = optim_cfg.BASE_LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    optim_dict = optim_cfg.get(optim_cfg.TYPE, dict())
    
    # specific lr settings for parts of params
    for i, key_words in enumerate(optim_dict['param_keys']):
        params = []
        mult_type = optim_dict['mult_types'][i]
        mult = optim_dict['mults'][i]
        for name, param in model.named_parameters(): 
            if key_words in name:
                all_spec_names.append(name)
                params.append(param)
        param_groups.append({
            'params': params,
            'lr': base_lr * mult if 'lr' in mult_type else base_lr,
            'weight_decay': weight_decay * mult if 'weight_decay' in mult_type else weight_decay,
        })
    
    # universal settings for other params
    uni_params = [param for name, param in model.named_parameters() if name not in all_spec_names]
    param_groups.append({'params': uni_params})
    optim_dict.pop('mult_types') 
    optim_dict.pop('param_keys') 
    optim_dict.pop('mults') 
    
    return optim_dict, param_groups
    
      

def build_scheduler(sche_cfg, optimizer):
    name = sche_cfg.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **sche_cfg.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if sche_cfg.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(sche_cfg.CLIP_LR))
        scheduler = ClipLR(scheduler, min_lr=sche_cfg.CLIP_LR)

    return scheduler
