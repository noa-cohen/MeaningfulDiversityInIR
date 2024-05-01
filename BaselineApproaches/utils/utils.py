import os
import numpy as np
import torch


def set_seed(seed: int = 125) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_path(path: str, exclusive: bool = False, verbose: bool = True) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print('mkdir [{:s}] ...'.format(path))
    elif exclusive:
        print('Folder [{:s}] already exists. Exit...'.format(path))
        exit(1)


def load_pt(pt_path: str, feature_str: str):
    try:
        pt = torch.load(pt_path)
        features = pt[feature_str]
    except Exception:
        raise ValueError(f"Could not load {pt_path}")
    if len(features) == 0:
        raise RuntimeError("Empty features")
    return features
