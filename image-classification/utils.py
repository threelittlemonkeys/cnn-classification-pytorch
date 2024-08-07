import sys
import re
from time import time
from os.path import isfile
from parameters import *
from collections import defaultdict

def load_tkn_to_idx(filename):

    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):

    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):

    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        print(tkn, file = fo)
    fo.close()

def load_checkpoint(filename, model = None):

    print("loading %s" % filename)
    checkpoint = torch.load(filename, map_location = DEVICE)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):

    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving %s" % filename)
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def maskset(x):

    mask = x.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths

def idx_to_tkn(tkn_to_idx):

    return [x for x, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1])]

def f1(p, r):

    return 2 * p * r / (p + r) if p + r else 0
