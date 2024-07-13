import scipy.io as scio
import numpy as np


def read_channel(file_path):

    X = scio.loadmat(file_path)['channels']
    X_r, X_i = np.real(X), np.imag(X)

    X_ = np.concatenate((X_r.transpose(), X_i.transpose()), axis=1)

    return X_


def egc_calc(X):

    # X: (num_of_sample, 2 * num_of_ant) --> [X_r, X_i]

    num_of_ant = int(X.shape[1] / 2)
    X_r, X_i = X[:, :num_of_ant], X[:, num_of_ant:]
    norm_one = np.nansum(np.sqrt(np.power(X_r, 2) + np.power(X_i, 2)), axis=1)
    egc_avg = (1 / num_of_ant) * np.nanmean(np.power(norm_one, 2))

    return egc_avg