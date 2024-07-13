import os
import argparse
import random
import torch
import numpy as np
import scipy.io as scio

from train_model import train
from read_data import read_channel, egc_calc


random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='gpu_id')
    parser.add_argument(dest='user_group')
    args = parser.parse_args()

    options = {
        'gpu_idx': int(args.gpu_id),
        'num_ant': 16,
        'num_bits': 3,
        'user_group': int(args.user_group),
        'interf_factor': 74,
        'noise_variance': 0.0138,
        'pf_print': 10,
        'path_bs_3': './data/O1_60_BS3_800_1200.mat',
        'save_freq': 50000,

        'interf_ant': 16,
        'interf_cb_size': 16,
        'interf_beam_path': "../BS4/beams/"
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 20000,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 8192,
        'minibatch_size': 1024,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')
    
    options['stats_path'] = os.path.join(os.getcwd(), 'stats', 'beam' + str(options['user_group']))
    if not os.path.exists(options['stats_path']):
        os.makedirs(options['stats_path'])

    # %% -------------- read only one cluster --------------

    BS3_UE = read_channel(options['path_bs_3'])
    user_group = scio.loadmat('./data/bs3_user_group.mat')['user_group']
    clus = user_group[0, options['user_group']].squeeze().tolist()
    ch = BS3_UE[clus, :]  # (num_of_sample, 2 * num_of_ant)

    egc_gain = egc_calc(ch)
    options['egc_gain'] = egc_gain

    H = scio.loadmat('./data/O1_60_BS3_BS4_new.mat')['H12']
    H_r, H_i = np.real(H), np.imag(H)
    H_interf = np.concatenate((H_r, H_i), axis=1) * options['interf_factor']
    options['H_interf'] = torch.from_numpy(H_interf).float()

    RF_cb = scio.loadmat('./data/steering_codebook_ant_16_size_16.mat')['W_UPA']
    RF_cb_r, RF_cb_i = np.real(RF_cb), np.imag(RF_cb)
    RF_cb = np.concatenate((RF_cb_r, RF_cb_i), axis=1)  # (num_ant, 2 * num_beams)
    options['RF_cb'] = torch.from_numpy(RF_cb).float()

    # %% ---------------------------------------------------

    # %% -------------- Quantization settings --------------

    options['num_ph'] = 2 ** options['num_bits']
    options['multi_step'] = torch.from_numpy(
        np.linspace(int(-(options['num_ph'] - 2) / 2),
                    int(options['num_ph'] / 2),
                    num=options['num_ph'],
                    endpoint=True)).type(dtype=torch.float32).reshape(1, -1)
    options['pi'] = torch.tensor(np.pi)
    options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
    options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)

    # %% ---------------------------------------------------

    for beam_id in [options['user_group']]:
        train(ch, options, train_opt, beam_id)
