import os
import torch
import random
import argparse
import numpy as np


def get_basic_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein-FV3")
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args([])


def get_dataset_params(args):
    if args.datasets == 'PLUSVein-FV3':
        args.split = '3:2'
        args.classes = 360
        args.pad_height_width = 736
        args.data_type = ['LED', 'LASER']
        args.data_root = '/mnt/c/Users/msp/Documents/git-repo/plus/PLUSVein-FV3/PLUSVein-FV3-ROI_combined/ROI'
        args.root_model = './checkpoint/PLUSVein-FV3'
        args.annot_file = 'annotations_plusvein.pkl'
    return args


def get_optim_params(args):
    if args.optim == 'adamw':
        args.lr = 2e-3
        args.weight_decay = 1e-2
    if args.optim == 'sgd':
        args.lr = 1e-1
        args.momentum = 0.9
        args.weight_decay = 2e-4
        
    if 'cosine' in args.scheduler:
        args.T_max = 16
        args.eta_min = 1e-6
        
    if 'ReduceLROnPlateau' in args.scheduler:
        args.factor = 0.9
        args.patience = 10
        args.verbose = True
    return args

def get_unlinkability_params(args):
    args.omega = 1.0
    args.n_bins = 100
    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_all_params():
    args = get_basic_params()
    args = get_optim_params(args)
    args = get_dataset_params(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
