import scipy.io as scio
import numpy as np
import torch


def cMUL(X, Y):

    # complex matrix multiplication Z = XY, where X and Y are complex matrices
    # Input: X = [X_r, X_i], Y = [Y_r, Y_i]; Output: Z = [Z_r, Z_i]

    half_x = int(X.size(1) / 2)
    half_y = int(Y.size(1) / 2)

    X_r = X[:, :half_x]
    X_i = X[:, half_x:]
    Y_r = Y[:, :half_y]
    Y_i = Y[:, half_y:]

    Z_r = X_r @ Y_r - X_i @ Y_i
    Z_i = X_r @ Y_i + X_i @ Y_r

    Z = torch.cat((Z_r, Z_i), dim=1)

    return Z


def ctranspose(X):

    # X: [X_r, X_i]
    # X^H: [X_r^T, -X_i^T]

    half = int(X.size(1) / 2)
    X_r, X_i = X[:, :half], X[:, half:]

    X_r_t = torch.t(X_r)
    X_i_t = torch.t(X_i)
    X_t = torch.cat((X_r_t, -X_i_t), dim=1)

    return X_t


def gain_pat_codebook(W, F, H):

    # Input: W: [W_r, W_i]; F: [F_r, F_i]; H: [H_r, H_i]
    # Output: abs(W^HHF)^2

    Z = cMUL(ctranspose(W), H)
    P = cMUL(Z, F)

    half = int(P.size(1) / 2)
    P = torch.abs(P[:, :half]) ** 2 + torch.abs(P[:, half:]) ** 2

    return P


def user_power_association(ch_set_1, ch_set_2):

    num_of_user = ch_set_1.shape[0]
    num_of_ant = int(ch_set_1.shape[1] / 2)

    H_1_r, H_1_i = ch_set_1[:, :num_of_ant], ch_set_1[:, num_of_ant:]
    pow_vec_1 = np.nansum(np.power(H_1_r, 2) + np.power(H_1_i, 2), axis=1)

    H_2_r, H_2_i = ch_set_2[:, :num_of_ant], ch_set_2[:, num_of_ant:]
    pow_vec_2 = np.nansum(np.power(H_2_r, 2) + np.power(H_2_i, 2), axis=1)

    user_set_1 = np.arange(num_of_user)[np.greater(pow_vec_1, pow_vec_2)]
    user_set_2 = np.setdiff1d(np.arange(num_of_user), user_set_1)

    return user_set_1, user_set_2


def cb_eval(thetas, X):

    # thetas: (num_of_beam, num_of_ant)
    # X: [X_r, X_i] --> (num_of_sample, 2 * num_of_ant)

    num_of_beam = thetas.shape[0]
    num_of_ant = int(X.shape[1] / 2)

    X = X.transpose()

    W_r = np.cos(thetas)  # (num_beam, num_ant)
    W_i = np.sin(thetas)  # (num_beam, num_ant)
    W1 = np.concatenate((W_r, W_i), axis=1)  # (num_beam, 2 * num_ant)
    W2 = np.concatenate((-W_i, W_r), axis=1)  # (num_beam, 2 * num_ant)
    W = (1 / np.sqrt(num_of_ant)) * np.concatenate((W1, W2), axis=0)  # (2 * num_beam, 2 * num_ant)
    assert X.shape[0] == W.shape[1], 'channel dimensions are not acceptable'  # The assumption is that X is a column vector
    A = np.matmul(W, X)  # (2 * num_beam, *)
    mask = np.eye(num_of_beam)
    power = A[:num_of_beam, :] ** 2 + A[num_of_beam:, :] ** 2  # (num_beam, *), welcome to the real domain
    idxs = power.argmax(axis=0)  # (*,)
    mask = mask[idxs]  # (*, num_beam)
    best_pow_vec = np.nansum(power * mask.transpose(), axis=0)  # (*,)
    avg_gain = np.nanmean(best_pow_vec)

    return avg_gain


if __name__ == '__main__':

    H12 = scio.loadmat('./O1_60_BS3_BS4.mat')['H12']
    H21 = scio.loadmat('./O1_60_BS4_BS3.mat')['H21']

    H12_r, H12_i = np.real(H12), np.imag(H12)
    H21_r, H21_i = np.real(H21), np.imag(H21)

    H12_ = np.concatenate((H12_r, H12_i), axis=1)
    H21_ = np.concatenate((H21_r, H21_i), axis=1)

    W_UPA = scio.loadmat('./steering_codebook_ant_16_size_64.mat')['W_UPA']
    W_UPA_r, W_UPA_i = np.real(W_UPA), np.imag(W_UPA)
    W_UPA_ = np.concatenate((W_UPA_r, W_UPA_i), axis=1)

    H12_tensor = torch.from_numpy(H12_).float()
    H21_tensor = torch.from_numpy(H21_).float()
    W_UPA_tensor = torch.from_numpy(W_UPA_).float()

    # Pat = gain_pat_codebook(W_UPA_tensor, W_UPA_tensor, H12_tensor)

    W = scio.loadmat('./test_data.mat')['W']
    F = scio.loadmat('./test_data.mat')['F']
    H = scio.loadmat('./test_data.mat')['H']
    Z = scio.loadmat('./test_data.mat')['Z']

    W_r, W_i = np.real(W), np.imag(W)
    W = np.concatenate((W_r, W_i), axis=1)
    W = torch.from_numpy(W).float()

    F_r, F_i = np.real(F), np.imag(F)
    F = np.concatenate((F_r, F_i), axis=1)
    F = torch.from_numpy(F).float()

    H_r, H_i = np.real(H), np.imag(H)
    H = np.concatenate((H_r, H_i), axis=1)
    H = torch.from_numpy(H).float()

    Z_r, Z_i = np.real(Z), np.imag(Z)
    Z = np.concatenate((Z_r, Z_i), axis=1)
    Z = torch.from_numpy(Z).float()

    Z_TEST = gain_pat_codebook(W, F, H)

    error = torch.sum(torch.pow(Z - Z_TEST, 2))

    pp = 1

