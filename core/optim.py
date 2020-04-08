import torch
import sys
from core.tools import module_classes_to_dict

# ------------------------------------------------------------------------------------
# Export PyTorch optimizer
# ------------------------------------------------------------------------------------
_this = sys.modules[__name__]
_optimizer_classes = module_classes_to_dict(torch.optim, exclude_classes="Optimizer")
for name, constructor in _optimizer_classes.items():
    setattr(_this, name, constructor)
__all__ = _optimizer_classes.keys()
