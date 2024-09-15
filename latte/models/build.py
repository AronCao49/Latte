from latte.models.xmuda_arch import Net2DSeg, Net3DSeg, ModalAttn, Domain_CLF, AdversarialNetwork
from latte.models.metric import SegIoU, SegAccuracy
from latte.models.hr_module import HR_Block, CR_Block, Dense_2D
# from latte.models.hr_module_fast import HR_Block, CR_Block, Dense_2D


def build_model_2d(cfg):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD,
                     da_method=cfg.TRAIN.DA_METHOD,
                     output_all=True
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='iou_2d')
    return model, train_metric

def build_model_3d(cfg):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD,
                     da_method=cfg.TRAIN.DA_METHOD,
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='iou_3d')
    return model, train_metric

def build_model_attn(cfg, inplanes):
    model = ModalAttn(num_classes=cfg.MODEL_3D.NUM_CLASSES, inplanes=inplanes, dropout=0.7, mode="cat")
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='iou_xm')
    return model, train_metric

def build_model_domain_clf(cfg):
    model = Domain_CLF()
    train_metric = SegAccuracy()
    return model, train_metric

def build_model_cdan(cfg):
    model = AdversarialNetwork()
    train_metric = SegAccuracy()
    return model, train_metric

def build_model_hrcr(cfg):
    try:
        resize = cfg.DATASET_TARGET.get(cfg.DATASET_TARGET.TYPE)['resize']
    except:
        resize = None
    hr_cfg = cfg.get("HR_BLOCK", dict())
    cr_cfg = cfg.get("CR_BLOCK", dict())
    model_hr = HR_Block(resize=resize, **hr_cfg)
    model_cr = CR_Block(**cr_cfg)
    
    return model_hr, model_cr

def build_model_ds2d(cfg):
    ds_cfg = cfg.get("Dense_2D", dict())
    model_ds2d = Dense_2D(**ds_cfg)
    
    return model_ds2d