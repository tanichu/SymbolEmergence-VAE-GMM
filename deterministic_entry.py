"""Opt-in deterministic runtime configuration; pass remaining flags to main.py."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import random
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if '--seed' in sys.argv:
    random.seed(int(sys.argv[sys.argv.index('--seed')+1]))
else:
    random.seed(2)  # main.py's documented default
import main
if __name__ == '__main__':
    main.main()
