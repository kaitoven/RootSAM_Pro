import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def set_absolute_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoggerEngine:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_file = os.path.join(log_dir, "training.log")

    def log(self, step, metrics_dict, phase="Train"):
        log_str = f"[{phase}] Epoch/Step: {step} | "
        for k, v in metrics_dict.items():
            self.writer.add_scalar(f"{phase}/{k}", v, step)
            if isinstance(v, float):
                log_str += f"{k}: {v:.4f}  "
            else:
                log_str += f"{k}: {v}  "
        print(log_str)
        with open(self.log_file, "a") as f:
            f.write(log_str + "\n")