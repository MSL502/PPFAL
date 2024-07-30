import torch
import random
import os
import numpy as np

def seed_setting(seed):

    random.seed(seed)
    # os.environ['']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

