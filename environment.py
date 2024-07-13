import os
import torch
import numpy as np
import pdb

from utils import cMUL


class envCB:

    def __init__(self, ch, num_ant, num_bits, idx, options):

        self.idx = idx
        self.num_ant = num_ant
        self.num_bits = num_bits
        self.cb_size = 2 ** self.num_bits
        self.codebook = self.codebook_gen()
        self.ch = torch.from_numpy(ch).float().cuda()
        self.H_interf = options['H_interf']
        self.RF_cb = options['RF_cb']
        self.state = torch.zeros((1, self.num_ant)).float().cuda()
        self.bf_vec = self.init_bf_vec()
        self.previous_gain = 0
        self.previous_sir = 0
        self.previous_mean = 0
        self.previous_std = 0
        self.previous_gain_pred = 0
        self.th_step = 0.01
        self.threshold = torch.tensor([0]).float().cuda()
        self.threshold_i = torch.tensor([100]).float().cuda()
        self.threshold_sir = torch.tensor([-100]).float().cuda()
        self.count = 1
        self.record_freq = 10
        self.record_decay_th = 1000
        self.achievement = torch.tensor([0]).float().cuda()
        self.achievement_sir = torch.tensor([0]).float().cuda()
        self.gain_record = [np.array(-100)]
        self.N_count = 1
        self.best_bf_vec = self.init_best()
        self.options = options

    def step(self, input_action):
        self.state = input_action
        reward, bf_gain, sir = self.reward_fn(self.state, 0)
        terminal = 0
        return self.state.clone(), reward, bf_gain, sir, terminal

    def reward_fn(self, ph_vec, flag):
        bf_vec = self.phase2bf(ph_vec)
        bf_gain_t, bf_gain_i_mean, bf_gain_i_std = self.sir_calc(bf_vec)
        noise = torch.tensor(self.options['noise_variance']).float().cuda()
        snr = torch.tensor(10.).float().cuda() * torch.log10(bf_gain_t / noise)
        sir = torch.tensor(10.).float().cuda() * torch.log10(bf_gain_t / (bf_gain_i_mean + noise))

        if bf_gain_t > self.previous_gain and bf_gain_i_mean < self.previous_mean:
            reward = np.array([1]).reshape((1, 1))
            if bf_gain_t > self.threshold and bf_gain_i_mean < self.threshold_i:
                self.threshold_modif(ph_vec, bf_gain_t, sir, bf_gain_i_mean)
        else:
            reward = np.array([-1]).reshape((1, 1))

        if flag:
            self.previous_gain = bf_gain_t
            self.previous_sir = sir
            self.previous_mean = bf_gain_i_mean
            self.previous_std = bf_gain_i_std
        return reward, bf_gain_t, sir

    def get_reward(self, input_action):
        inner_state = input_action

        # Quantization Processing
        mat_dist = torch.abs(inner_state.reshape(self.num_ant, 1) - self.options['ph_table_rep'])
        action_quant = self.options['ph_table_rep'][range(self.num_ant), torch.argmin(mat_dist, dim=1)].reshape(1, -1)

        reward, bf_gain_t, sir = self.reward_fn(action_quant, 1)
        self.count += 1
        return reward, bf_gain_t, sir, action_quant.clone(), action_quant.clone()

    def threshold_modif(self, ph_vec, bf_gain, sir, bf_gain_i_mean):
        self.achievement = bf_gain
        self.achievement_sir = sir
        self.gain_recording(ph_vec, self.idx)
        self.threshold = bf_gain
        self.threshold_i = bf_gain_i_mean
        self.threshold_sir = sir

    def phase2bf(self, ph_vec):
        bf_vec = torch.zeros((1, 2 * self.num_ant)).float().cuda()
        for kk in range(self.num_ant):
            bf_vec[0, 2*kk] = torch.cos(ph_vec[0, kk])
            bf_vec[0, 2*kk+1] = torch.sin(ph_vec[0, kk])
        norm_factor = np.sqrt(np.divide(1, self.num_ant))
        bf_vec = torch.tensor(norm_factor).float().cuda() * bf_vec
        return bf_vec

    def sir_calc(self, bf_vec):
        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)

        ch_r = self.ch[:, :self.num_ant].clone()
        ch_i = self.ch[:, self.num_ant:].clone()
        bf_gain_t_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_t_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_t_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_t_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_t_r = (bf_gain_t_1 + bf_gain_t_2) ** 2
        bf_gain_t_i = (bf_gain_t_3 - bf_gain_t_4) ** 2
        bf_gain_pattern_t = bf_gain_t_r + bf_gain_t_i
        bf_gain_t = torch.mean(bf_gain_pattern_t)

        interf_cb = torch.zeros(self.options['interf_ant'], 2 * self.options['interf_cb_size']).float().cuda()
        for ii in range(self.options['interf_cb_size']):
            interf_beam = self.get_beam(self.options['interf_beam_path'] + 'beams_' + str(ii) + '_max.txt')
            # pdb.set_trace()
            interf_cb[:, ii] = interf_beam[:, 0]
            interf_cb[:, ii + self.options['interf_cb_size']] = interf_beam[:, 1]

        self.RF_cb = interf_cb

        Z = cMUL(self.H_interf, self.RF_cb)
        bf = torch.cat((bf_r, -bf_i), dim=1)

        bf_gain_i = cMUL(bf, Z)
        RF_cb_size = int(self.RF_cb.size(1) / 2)

        bf_gain_i_r = torch.abs(bf_gain_i[0:1, :RF_cb_size]) ** 2
        bf_gain_i_i = torch.abs(bf_gain_i[0:1, RF_cb_size:]) ** 2
        bf_gain_pattern_i = bf_gain_i_r + bf_gain_i_i

        bf_gain_i_mean = torch.mean(bf_gain_pattern_i)
        bf_gain_i_std = torch.std(bf_gain_pattern_i)

        return bf_gain_t, bf_gain_i_mean, bf_gain_i_std

    def gain_recording(self, bf_vec, idx):

        new_gain = torch.Tensor.cpu(self.achievement).detach().numpy().reshape((1, 1))
        new_sir = torch.Tensor.cpu(self.achievement_sir).detach().numpy().reshape((1, 1))
        new_record = np.concatenate((new_gain, new_sir), axis=1)
        bf_print = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)

        if new_sir > max(self.gain_record):
            self.gain_record.append(new_sir)
            self.best_bf_vec = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
            if os.path.exists('beams/beams_' + str(idx) + '_max.txt'):
                with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, new_record, fmt='%.2f', delimiter=',')
                with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            else:
                np.savetxt('beams/beams_' + str(idx) + '_max.txt', new_record, fmt='%.2f', delimiter=',')
                # with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                #     np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')

    def codebook_gen(self):
        angles = np.linspace(0, 2 * np.pi, self.cb_size, endpoint=False)
        cb = np.exp(1j * angles)
        codebook = torch.zeros((self.cb_size, 2)) # shape of the codebook
        for ii in range(cb.shape[0]):
            codebook[ii, 0] = torch.tensor(np.real(cb[ii]))
            codebook[ii, 1] = torch.tensor(np.imag(cb[ii]))
        return codebook

    def init_bf_vec(self):
        bf_vec = torch.empty((1, 2 * self.num_ant))
        bf_vec[0, ::2] = torch.tensor([1])
        bf_vec[0, 1::2] = torch.tensor([0])
        bf_vec = bf_vec.float().cuda()
        return bf_vec

    def init_best(self):
        ph_book = np.linspace(-np.pi, np.pi, 2 ** self.num_bits, endpoint=False)
        ph_vec = np.array([[ph_book[np.random.randint(0, len(ph_book))] for ii in range(self.num_ant)]])
        bf_complex = np.exp(1j * ph_vec)
        bf_vec = np.empty((1, 2 * self.num_ant))
        for kk in range(self.num_ant):
            bf_vec[0, 2*kk] = np.real(bf_complex[0, kk])
            bf_vec[0, 2*kk+1] = np.imag(bf_complex[0, kk])
        return bf_vec

    def update_interf_beam(self, num_ant, num_beam):

        phase_collec = np.linspace(0, 2 * np.pi, 2 ** self.num_bits, endpoint=False)

        size_interf_codebook = num_beam
        num_of_ant_interf_bs = num_ant

        interf_codebook = np.zeros((num_of_ant_interf_bs, 2 * size_interf_codebook))

        for ii in range(size_interf_codebook):
            angles = np.random.choice(phase_collec, size=(num_of_ant_interf_bs,), replace=True)
            beam = (1 / np.sqrt(num_of_ant_interf_bs)) * np.exp(1j * angles)
            beam_r, beam_i = np.real(beam), np.imag(beam)
            interf_codebook[:, ii] = beam_r
            interf_codebook[:, ii + size_interf_codebook] = beam_i

        # pdb.set_trace()
        return torch.from_numpy(interf_codebook).float().cuda()

    def get_beam(self, path):

        # path = "./test.txt"

        if os.path.exists(path):
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    phases = np.fromstring(last_line.replace("\n", ""), sep=',')
                    beam_complex = (1 / np.sqrt(self.options['interf_ant'])) * np.exp(1j * phases)
                    beam = np.zeros((self.options['interf_ant'], 2))
                    beam[:, 0] = np.real(beam_complex)
                    beam[:, 1] = np.imag(beam_complex)
                    return torch.from_numpy(beam).float().cuda()
                else:
                    return self.update_interf_beam(self.options['interf_ant'], 1)
        else:
            return self.update_interf_beam(self.options['interf_ant'], 1)
