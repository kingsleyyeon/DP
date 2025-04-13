import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import random
import sys
import pickle
import os
import pandas as pd


# Set random seed for reproducibility.
np.random.seed(13)

######################################
# Helper Functions 
######################################
def clip_C(val, C):
    return np.clip(val, -C, C)

def project_matrix(M, radius, dp_enabled=True):
    if dp_enabled:
        fro_norm = np.linalg.norm(M, 'fro')
        if fro_norm > radius:
            return (radius / fro_norm) * M
        return M
    else:
        return M

def sample_unit_sphere(d):
    x = np.random.randn(d)
    return x / np.linalg.norm(x)

def sample_L_unit_sphere(d, L):
    X = np.random.randn(L, d)
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / X_norms

def compute_Ztilde(x_query, xs, ys, L, G, C, dp_enabled=True):
    """
    Compute the summary statistic Z_tilde for a query point.

    Z_tilde = Pi_G((1/L) * outer(x_query, sum_i clip_C(ys[i]) * xs[i]))

    Parameters:
        x_query (np.ndarray): Query vector of shape (D,).
        xs (np.ndarray): Data vectors of shape (L, D).
        ys (np.ndarray): Labels of shape (L,).
        L (int): Number of samples.
        G (float): Frobenius norm clipping threshold.
        C (float): Label clipping threshold.
        dp_enabled (bool): Whether to apply clipping and projection.

    Returns:
        np.ndarray: Projected matrix Z_tilde of shape (D, D).
    """
    ys_clipped = np.clip(ys, -C, C) if dp_enabled else ys
    v = np.sum(ys_clipped[:, None] * xs, axis=0)
    Z = (1.0 / L) * np.outer(x_query, v)
    return project_matrix(Z, G, dp_enabled)


def compute_Ztilde_vectorized(xq, xs, ys, L, G, C, dp_enabled=True):
    """
    Vectorized computation of the summary statistic for a batch of prompts.
      xq: shape (N, D)
      xs: shape (N, L, D)
      ys: shape (N, L)
    Returns:
      Z: shape (N, D, D)
    """
    if dp_enabled:
        ys_clipped = np.clip(ys, -C, C)
    else:
        ys_clipped = ys
    # Compute v for each prompt: shape (N, D)
    v = np.sum(ys_clipped[..., None] * xs, axis=1)
    # Compute outer product for each prompt: shape (N, D, D)
    Z = (1.0 / L) * (xq[:, :, None] * v[:, None, :])
    if dp_enabled:
        fro_norm = np.linalg.norm(Z, axis=(1,2))
        scaling = np.where(fro_norm > G, G / fro_norm, 1.0)
        Z = Z * scaling[:, None, None]
    return Z



def dp_noise(shape, noise_std):
    return np.random.normal(0, noise_std, size=shape)

def generate_prompt(D, L):
    xs = sample_L_unit_sphere(D, L)
    x_query = sample_unit_sphere(D)
    w_true = np.random.randn(D)
    ys = xs @ w_true
    y_query = x_query @ w_true
    return xs, x_query, ys, y_query

def build_dataset(N, D, L):
    xs_all, xq_all, ys_all, yq_all = [], [], [], []
    for _ in range(N):
        xs, xq, ys, yq = generate_prompt(D, L)
        xs_all.append(xs)
        xq_all.append(xq)
        ys_all.append(ys)
        yq_all.append(yq)
    return np.array(xs_all), np.array(xq_all), np.array(ys_all), np.array(yq_all)


def build_bad_dataset(D, L, mu, alpha):
    xs_all, xq_all, ys_all, yq_all = build_dataset(1, D, L) 
    xs_all = np.array([x + mu for x in xs_all])
    xq_all = np.array([x + mu for x in xq_all])
    ys_all = np.array([y + alpha for y in ys_all])
    yq_all = np.array([y + alpha for y in yq_all])
    
    return xs_all, xq_all, ys_all, yq_all

def train_algorithm2(xs_all, xq_all, ys_all, yq_all,
                     D, N, L,
                     C, R, G, lam, eta0, T, epsilon, delta,
                     sigma_multiplier, dp_enabled):
    """
    Trains "Algorithm 2" from the paper.

    Parameters:
        xs_all, xq_all, ys_all, yq_all: dataset tensors
        D (int): Dimensionality
        N (int): Number of prompts
        L (int): Number of training tokens per prompt
        C, R, G, lam, eta0: model hyperparameters
        T (int): Number of training steps
        epsilon, delta: DP parameters
        sigma_multiplier (float): scales DP noise
        dp_enabled (bool): whether DP is active

    Returns:
        train_losses (list): MSE per iteration
        Gamma (np.ndarray): final (D, D) matrix
        Ztilde_ls (np.ndarray): summary matrices (N, D, D)
        noise_std (float): DP noise standard deviation
    """

    # Precompute Z_tilde matrices
    Ztilde_ls = np.array([
        compute_Ztilde(xq_all[k], xs_all[k], ys_all[k], L, G, C, dp_enabled)
        for k in range(N)
    ])  # shape (N, D, D)

    # Differential Privacy noise scale
    if dp_enabled:
        sigma_lower_bound = 2.0 * G * (C + R * G)
        sigma = sigma_lower_bound * sigma_multiplier
        denom = (epsilon**2) * (N**2)
        noise_variance = (2.0 * (eta0**2) * (T**2) * (sigma**2) * np.log(1.25 * T / delta)) / denom
        noise_std = np.sqrt(noise_variance)
    else:
        noise_std = 0.0

    # Initialize Gamma
    Gamma = np.random.randn(D, D) * 0.1
    Gamma = project_matrix(Gamma, R, dp_enabled)

    train_losses = []

    for t in tqdm(range(T), desc="Training steps", ncols=100, position=0):
        # Predict y for each prompt
        y_preds = np.einsum('nij,ij->n', Ztilde_ls, Gamma)  # shape (N,)
        y_clips = np.clip(yq_all, -C, C) if dp_enabled else yq_all
        errs = y_preds - y_clips

        # Gradient update
        grad_sum = np.einsum('n,nij->ij', errs, Ztilde_ls) / N
        grad_mat = grad_sum + 2.0 * lam * Gamma

        # DP noise
        noise = dp_noise(Gamma.shape, noise_std)

        # Update Gamma
        Gamma = Gamma - eta0 * grad_mat + noise
        Gamma = project_matrix(Gamma, R, dp_enabled)

        # MSE loss
        mse_avg = np.mean((y_preds - yq_all) ** 2)
        train_losses.append(mse_avg)

    return train_losses, Gamma, Ztilde_ls, noise_std



def run_experiment(N, L, D, B, mu, alpha, epsilon, delta, sigma_multiplier):

    # Precompute constants
    lam = 0.01
    C = np.sqrt(2 * np.log(N * L))
    R = 0.1 * (C ** 2) * np.sqrt(N / L)
    G = (C / np.sqrt(L)) * (1 + (np.log(N) / (D ** 2)) ** 0.25)
    eta0 = 2 * lam / (3 * (lam + G ** 2) ** 2)
    T = int(np.log(N))

    # Initialize error arrays
    err_ridge_array = []
    err_dp_array = []

    for _ in tqdm(range(B), desc="B many iterations", ncols=100, position=0, leave=False):
        # Generate clean train/test data and bad point
        xs_train, xq_train, ys_train, yq_train = build_dataset(N, D, L)
        xs_bad, xq_bad, ys_bad, yq_bad = build_bad_dataset(D, L, mu, alpha)
        xs_test, xq_test, ys_test, yq_test = build_dataset(500, D, L)

        # --- Train DP model on clean data ---
        _, gamma_dp_good, _, _ = train_algorithm2(
            xs_train, xq_train, ys_train, yq_train,
            D, N, L, C, R, G, lam, eta0, T,
            epsilon, delta, sigma_multiplier, dp_enabled=True
        )

        # --- Ridge oracle on clean data ---
        Z_good = np.array([
            compute_Ztilde(xq_train[k], xs_train[k], ys_train[k], L, G, C, dp_enabled=False)
            for k in range(N)
        ])
        d1, d2 = Z_good.shape[1:]
        D_total = d1 * d2
        A = lam * np.eye(D_total) + (1/N) * Z_good.reshape(N, -1).T @ Z_good.reshape(N, -1)
        b_vec = yq_train.reshape(N, -1)
        b = (1/N) * Z_good.reshape(N, -1).T @ b_vec
        gamma_vec = np.linalg.solve(A, b)
        Gamma_star_good = gamma_vec.reshape(d1, d2)

        # --- Evaluate on test set ---
        Z_test_all = compute_Ztilde_vectorized(xq_test, xs_test, ys_test, L, G, C, dp_enabled=False)
        y_hat_dp_good = np.sum(gamma_dp_good * Z_test_all, axis=(1, 2))
        y_hat_ridge_good = np.sum(Gamma_star_good * Z_test_all, axis=(1, 2))

        # --- Inject bad point ---
        i = random.randint(0, N - 1)
        xs_temp = xs_train.copy()
        xq_temp = xq_train.copy()
        ys_temp = ys_train.copy()
        yq_temp = yq_train.copy()

        xs_temp[i] = xs_bad.squeeze()
        xq_temp[i] = xq_bad.squeeze()
        ys_temp[i] = ys_bad.squeeze()
        yq_temp[i] = yq_bad.squeeze()

        # --- Train DP on corrupted data ---
        _, gamma_dp_bad, _, _ = train_algorithm2(
            xs_temp, xq_temp, ys_temp, yq_temp,
            D, N, L, C, R, G, lam, eta0, T,
            epsilon, delta, sigma_multiplier, dp_enabled=True
        )
        y_hat_dp_bad = np.sum(gamma_dp_bad * Z_test_all, axis=(1, 2))

        # --- Ridge oracle on corrupted data ---
        Z_bad = np.array([
            compute_Ztilde(xq_temp[k], xs_temp[k], ys_temp[k], L, G, C, dp_enabled=False)
            for k in range(N)
        ])
        A_bad = lam * np.eye(D_total) + (1/N) * Z_bad.reshape(N, -1).T @ Z_bad.reshape(N, -1)
        b_bad = (1/N) * Z_bad.reshape(N, -1).T @ yq_temp.reshape(N, -1)
        gamma_vec_bad = np.linalg.solve(A_bad, b_bad)
        Gamma_star_bad = gamma_vec_bad.reshape(d1, d2)
        y_hat_ridge_bad = np.sum(Gamma_star_bad * Z_test_all, axis=(1, 2))

        # --- Accumulate error ---
        err_ridge_array.append(np.mean((y_hat_ridge_good - y_hat_ridge_bad) ** 2))
        err_dp_array.append(np.mean((y_hat_dp_good - y_hat_dp_bad) ** 2))

    return np.mean(err_ridge_array), np.mean(err_dp_array)


# Global parameters.
epsilon = 0.5
delta = 1e-2
sigma_multiplier = 1
N = 5000
L = 500
D = 5
B = 500
mu = 1

# Get p and c from command line
p = float(sys.argv[1])
c = float(sys.argv[2])
alpha = c * (N ** p)

# Run experiment
try:
    err_ridge, err_dp = run_experiment(
        N=N, L=L, D=D, B=B, mu=mu, alpha=alpha,
        epsilon=epsilon, delta=delta, sigma_multiplier=sigma_multiplier
    )

    os.makedirs("results", exist_ok=True)
    result_df = pd.DataFrame([{
        "p": p,
        "c": c,
        "alpha": alpha,
        "err_ridge": err_ridge,
        "err_dp": err_dp
    }])
    result_df.to_csv(f"results/result_p{p}_c{c}.csv", index=False)

except Exception as e:
    os.makedirs("results", exist_ok=True)
    with open(f"results/error_p{p}_c{c}.log", "w") as f:
        f.write(f"Error for p={p}, c={c}:\n{str(e)}\n")