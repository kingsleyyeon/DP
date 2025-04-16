import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm
import math
import pandas as pd
import sys
import os

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

    for t in range(T):
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

def run_experiment(N, epsilon, n_simulations=500):
    """
    For a single N and epsilon, run the training and evaluation 
    n_simulations times and average the excess risks.

    Returns:
      - avg_excess_risk_no_dp: average excess risk (non-DP) over n_simulations
      - avg_excess_risk_dp: average excess risk (DP) over n_simulations
    """

    # Initialize lists to store excess risks
    excess_risks_no_dp = []
    excess_risks_dp = []

    for _ in tqdm(range(n_simulations), desc="Training", ncols=100, position=0):
        N_train = N
        N_test = 500
        D = int(N ** 0.5)
        L = int(N ** 0.5)
        lam = N / D
        C = np.sqrt(2 * np.log(N * L))
        R = (1 / lam) * (C ** 2) * np.sqrt(N / L)
        G = (C / np.sqrt(L)) * (1 + (np.log(N) / (D ** 2)) ** (1 / 4))
        eta0 = 0.1 * (2 * lam / (3 * ((lam + G ** 2) ** 2)))
        T = 5

        #print(f"1-eta0*lambda = {(1 - eta0 * lam)}")
        #print(f"N = {N}")
        #print(f"L = {L}")
        #print(f"D = {D}")
        #print(f"C = {C}")
        #print(f"R = {R}")
        #print(f"G = {G}")
        #print(f"T = {T}")
        #print(f"sigma={2.0 * G * (C + R * G)}")
        #print(f"lambda = {lam}")
        #print(f"step-size = {eta0}")

        xs_train, xq_train, ys_train, yq_train = build_dataset(N, D, L)
        xs_test, xq_test, ys_test, yq_test = build_dataset(N_test, D, L)

        # Train non-DP Algorithm 2
        losses_no_dp, Gamma_no_dp, Ztilde_ls, _ = train_algorithm2(
            xs_train, xq_train, ys_train, yq_train,
            D, N, L,
            C, R, G, lam, eta0, T,
            epsilon, delta,
            sigma_multiplier, dp_enabled=False
        )

        # Compute the local ridge oracle
        d1, d2 = Ztilde_ls.shape[1:3]
        D_total = d1 * d2
        Zmat = Ztilde_ls.reshape(N_train, D_total)
        ridge_lam = lam
        A = (N_train * ridge_lam) * np.eye(D_total) + Zmat.T @ Zmat
        b = Zmat.T @ yq_train
        L_chol = np.linalg.cholesky(A)
        v = np.linalg.solve(L_chol, b)
        gamma_vec = np.linalg.solve(L_chol.T, v)
        Gamma_star = gamma_vec.reshape(d1, d2)

        Z_test_all = compute_Ztilde_vectorized(xq_test, xs_test, ys_test, L, G, C, dp_enabled=False)
        y_hat_ridge = np.sum(Gamma_star * Z_test_all, axis=(1, 2))

        # DP training
        losses_dp, Gamma_dp, _, noise_std = train_algorithm2(
            xs_train, xq_train, ys_train, yq_train,
            D, N, L,
            C, R, G, lam, eta0, T,
            epsilon, delta,
            sigma_multiplier, dp_enabled=True
        )

        #print(f"dp-noise std = {noise_std}")

        y_hat_no_dp = np.sum(Gamma_no_dp * Z_test_all, axis=(1, 2))
        y_hat_dp = np.sum(Gamma_dp * Z_test_all, axis=(1, 2))

        mse_no_dp = np.mean((y_hat_no_dp - y_hat_ridge) ** 2)
        mse_dp = np.mean((y_hat_dp - y_hat_ridge) ** 2)

        excess_risks_no_dp.append(mse_no_dp)
        excess_risks_dp.append(mse_dp)

    avg_excess_risk_no_dp = np.mean(excess_risks_no_dp)
    avg_excess_risk_dp = np.mean(excess_risks_dp)

    return avg_excess_risk_no_dp, avg_excess_risk_dp


epsilon_lst = [0.01, 0.05, 0.1, 0.5, 1.0]
delta = 1e-2
sigma_multiplier = 1
n_simulations = 500
np.random.seed(1609)

# Parse N from command-line arguments
N = int(sys.argv[1])

# Run all experiments for this value of N
results = []
for epsilon in tqdm(epsilon_lst, desc=f"N={N}", ncols=100):
    avg_no_dp, avg_dp = run_experiment(N, epsilon, n_simulations=n_simulations)
    results.append({
        "N": N,
        "epsilon": epsilon,
        "avg_no_dp_excess": avg_no_dp,
        "avg_dp_excess": avg_dp
    })

# Save results to disk
os.makedirs("results_high_dim_vary_eps", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(f"results_high_dim_vary_eps/results_N{N}.csv", index=False)
