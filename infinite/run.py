import importlib
import logging
import random
import socket
import sys
import time

import hydra
import numpy as np
import sbi
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
import yaml
from metrics import compute_metrics_df
from omegaconf import DictConfig, OmegaConf
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import save_float_to_csv, save_tensor_to_csv

import cnre

# TODO make this, and the benchmark run function, call some internal thing so you don't need to write it twice.

dataset = cnre.joint.JointSampler(
    simulator,
    proposal,
    training_batch_size,
)
train_loader, val_loader = cnre.joint.get_endless_train_loader_and_new_valid_loader()
pass
