import json
from pathlib import Path, PurePosixPath
import torch
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Path, PurePosixPath)):
            return str(obj)
        elif hasattr(obj, '__name__') and hasattr(obj, '__code__'):
            # Object is a function
            return obj.__name__
        return super().default(obj)


def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    else:
        return np.log(x)
