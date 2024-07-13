import numpy as np
import scipy.io as scio

num_ant = 16
num_beam = 16  # change this
beam_id = list(range(num_beam))
results = np.empty((num_beam, num_ant))

path = './beams/'  # change this

for kk in beam_id:
    fname = 'beams_' + str(kk) + '_max.txt'
    with open(path + fname, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        results[kk, :] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

scio.savemat('Learned_codebook.mat',
             {'beams': results})
