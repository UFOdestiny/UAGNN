import argparse
import logging
import os
import platform
import sys

import numpy as np
import torch


def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)

    return logger


def get_config():
    if platform.system().lower() == 'linux':
        A_path='adj.npy'
        X_path='values_in.npy'
        checkpoint_path = "/"
        losses_p=""
        best_model_path = ''
        epochs = 4000
        batch_size = 64

        hidden_dim_s = 64
        hidden_dim_t = 16
        rank_s = 32
        rank_t = 8
        orders = 3
    else:
        A_path='/adj.npy'
        X_path='values_in.npy'
        checkpoint_path = ""
        losses_p=""
        best_model_path = ''
        epochs = 200
        batch_size = 16

        hidden_dim_s = 64
        hidden_dim_t = 32
        rank_s = 32
        rank_t = 16
        orders = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--A_path', type=str, default=A_path)
    parser.add_argument('--X_path', type=str, default=X_path)
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)
    parser.add_argument('--losses_p', type=str, default=losses_p)
    parser.add_argument('--best_model_path', type=str, default=best_model_path)
    parser.add_argument('--Xpath', type=str, default=X_path)
    parser.add_argument('--num_timesteps_input', type=int, default=12)
    parser.add_argument('--num_timesteps_output', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--hidden_dim_s', type=int, default=hidden_dim_s)
    parser.add_argument('--hidden_dim_t', type=int, default=hidden_dim_t)
    parser.add_argument('--rank_s', type=int, default=rank_s)
    parser.add_argument('--rank_t', type=int, default=rank_t)
    parser.add_argument('--orders', type=int, default=orders)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=2024)


    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default="stzinb")

    args = parser.parse_args()


    log_dir = './log/{}/'.format(args.model_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


