import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 64
IMG_WIDTH = 70
IMG_HEIGHT = 30
LEARNING_RATE = 1e-4
VERBOSE = False
EVAL_EVERY = 10
SAVE_EVERY = 10

CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility
# torch.cuda.set_device(0)

t_Tensor = lambda *x: torch.FloatTensor(*x).cuda() if CUDA else torch.FloatTensor(*x)
t_LongTensor = lambda *x: torch.LongTensor(*x).cuda() if CUDA else torch.LongTensor(*x)
t_stack = lambda *x: torch.stack(*x).cuda() if CUDA else torch.stack(*x)
t_zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros(*x)

NUM_DIGITS = 4 # number of digits to print
