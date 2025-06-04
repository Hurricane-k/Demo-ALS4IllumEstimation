import numpy as np
import torch
import torch.nn as nn
import json


def calc_ang_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = torch.from_numpy(pred).float()
    gt = torch.from_numpy(gt).float()
    norm_a = torch.norm(pred)
    norm_b = torch.norm(gt)
    norm_a = pred / (norm_a + 1e-9)
    norm_b = gt / (norm_b + 1e-9)
    dot_ab = norm_a @ norm_b
    dot_ab = torch.clamp(dot_ab, -0.999999, 0.999999)
    ae = torch.rad2deg(torch.acos(dot_ab))
    ae = float(ae)
    return ae


def angular_error_torch(pred: torch.Tensor,
                        gt: torch.Tensor,
                        average: bool = True) -> torch.Tensor:
    norm_a = torch.norm(pred, dim=-1, keepdim=True)
    norm_b = torch.norm(gt, dim=-1, keepdim=True)
    norm_a = pred / (norm_a + 1e-9)
    norm_b = gt / (norm_b + 1e-9)
    dot_ab = torch.sum(norm_a * norm_b, dim=-1)
    dot_ab = torch.clamp(dot_ab, -0.999999, 0.999999)
    ae = torch.rad2deg(torch.acos(dot_ab))
    if average:
        ae = torch.mean(ae)
    return ae

def metrics_torch(torch_ae):
    """ return a dict
    """
    errors = sorted(torch_ae)
    metrics = {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'trimean': (0.25*np.percentile(errors,25)+
                    0.50*np.percentile(errors,50)+
                    0.25*np.percentile(errors,75)),
        'bst25': np.mean(errors[:int(len(errors)*0.25)]),
        'wst25': np.mean(errors[int(len(errors)*0.75):]),
        'wst05': np.mean(errors[int(len(errors)*0.95):])
    }
    return metrics

# Custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


ce_loss = nn.CrossEntropyLoss(reduction='mean')


def classification_loss(pred, target):
    return ce_loss(pred, target)
