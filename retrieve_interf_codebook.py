import os
import time
import numpy as np
import torch


def get_beam(path):

    # path = "./test.txt"

    if os.path.exists(path):
        if os.path.isfile(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            if lines:
                last_line = lines[-1]
                beam = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)
                return beam
            else:
                return update_interf_beam(16, 1)
    else:
        raise ValueError("Something is wrong.")


def codebook_gen():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    cb = np.exp(1j * angles)
    codebook = torch.zeros((8, 2))  # shape of the phase codebook
    for ii in range(cb.shape[0]):
        codebook[ii, 0] = torch.tensor(np.real(cb[ii]))
        codebook[ii, 1] = torch.tensor(np.imag(cb[ii]))
    return codebook


def update_interf_beam(num_ant, num_beam):

    phase_collec = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    size_interf_codebook = num_beam
    num_of_ant_interf_bs = num_ant

    interf_codebook = np.zeros((num_of_ant_interf_bs, 2 * size_interf_codebook))

    for ii in range(size_interf_codebook):
        angles = np.random.choice(phase_collec, size=(num_of_ant_interf_bs,), replace=True)
        beam = (1 / np.sqrt(num_of_ant_interf_bs)) * np.exp(1j * angles)
        beam_r, beam_i = np.real(beam), np.imag(beam)
        interf_codebook[:, ii] = beam_r
        interf_codebook[:, ii + size_interf_codebook] = beam_i

    return interf_codebook


if __name__ == '__main__':

    # interf_beam = get_beam("./test.txt")
    # print(interf_beam)

    x = update_interf_beam(16, 16)
    y = get_beam("./test.txt")

    pp = 1
