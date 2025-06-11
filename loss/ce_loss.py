from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch

@OPENOCC_LOSS.register_module()
class CeLoss_1(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        # self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        """ ce_labels.shape: f-1, H, W"""
        ce_loss = F.cross_entropy(ce_inputs, ce_labels)
        return ce_loss

@OPENOCC_LOSS.register_module()
class CeLoss_2(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        # self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        """ ce_labels.shape: f-1, H, W"""
        ce_loss = F.cross_entropy(ce_inputs, ce_labels)
        return ce_loss

@OPENOCC_LOSS.register_module()
class CeLoss_3(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        ce_loss = F.cross_entropy(ce_inputs, ce_labels, label_smoothing=0.1)
        return ce_loss
    
@OPENOCC_LOSS.register_module()
class CeLoss_4(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        ce_loss = F.cross_entropy(ce_inputs, ce_labels, label_smoothing=0.1)
        return ce_loss

@OPENOCC_LOSS.register_module()
class CeLoss_t(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        # self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        """ ce_labels.shape: f-1, H, W"""
        ce_loss = F.cross_entropy(ce_inputs, ce_labels)
        return ce_loss

@OPENOCC_LOSS.register_module()
class CeLoss_d(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        # self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels): # (bs*f, CH, L), (bs*f, L)
        # output: -1, 1
        """ ce_labels.shape: f-1, H, W"""
        ce_loss = F.cross_entropy(ce_inputs, ce_labels)
        return ce_loss