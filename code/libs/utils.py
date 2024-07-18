import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
from .config import config
import glob
import os 
from loguru import logger

def get_device():
    s = config("DEVICE", default="", section="train")
    if s == "":
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda:0")
        else:
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(s)
    return DEVICE


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1.2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

def batch_pesq(clean, noisy, device):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score + 0.5) / 5
    return torch.FloatTensor(pesq_score).to(device)

def get_epoch(cp) -> int:
    return int(os.path.basename(cp).split(".")[0].split("_")[-1])

def read_cp(model: nn.Module, 
            dirname: str, 
            name1="model_bsrnn", 
            extension="ckpt",
            is_teacher=False
            ):
    checkpoints_model = []
    checkpoints_model = glob.glob(os.path.join(dirname, f"{name1}*.{extension}"))
    checkpoints_model += glob.glob(os.path.join(dirname, f"{name1}*.{extension}.best"))
    if is_teacher:
        assert len(checkpoints_model) > 0
        latest_model_path = max(checkpoints_model, key=get_epoch)
        epoch = get_epoch(latest_model_path)
        
        latest_model = torch.load(latest_model_path, map_location="cpu")
        model.load_state_dict(latest_model)
        logger.info("Found teacher model checkpoint {} with epoch {}".format(latest_model_path, epoch))
        return model, epoch

    elif len(checkpoints_model) == 0:
        return model, 0
    else:
        latest_model_path = max(checkpoints_model, key=get_epoch)
        epoch = get_epoch(latest_model_path)
        
        latest_model = torch.load(latest_model_path, map_location="cpu")
        model.load_state_dict(latest_model)
        logger.info("Found student model checkpoint {} with epoch {}".format(latest_model_path, epoch))
        return model, epoch

    
        
    
    