import argparse
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal, wishart
from scipy.special import logsumexp
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score as ari
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

import cnn_vae_module_mnist
from tool import calc_ari, cmx


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Symbol emergence based on VAE+GMM Example"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="input batch size for training",
    )
    parser.add_argument(
        "--vae-iter", type=int, default=50, metavar="V", help="number of VAE iteration"
    )
    parser.add_argument(
        "--mh-iter",
        type=int,
        default=50,
        metavar="M",
        help="number of M-H mgmm iteration",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=10,
        metavar="K",
        help="number of category for GMM module",
    )
    parser.add_argument(
        "--mode", type=int, default=-1, metavar="M", help="0:All reject, 1:ALL accept"
    )
    parser.add_argument(
        "--debug", type=bool, default=False, metavar="D", help="Debug mode"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=2, metavar="S", help="random seed")
    parser.add_argument("--mutual-iterations", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10000, help="MNIST smoke/subset size, not balanced selection")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--prior-scale", type=float, default=0.1)
    parser.add_argument("--paper-mnist", action="store_true", help="10k balanced, 100 VAE/MH epochs, 5 MI rounds, prior W=.05I")
    parser.add_argument("--disable-mi", action="store_true", help="one initial VAE/GMM block; different total budget from MI")
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--cache-prior', action='store_true')
    parser.add_argument('--gpu-images', action='store_true')
    parser.add_argument('--epoch-sync', action='store_true')
    parser.add_argument('--paired-manifest', default=None, help='Fixed RGB64 paired-image manifest; separate from MNIST')

    args = parser.parse_args()
    if args.gpu_images and not args.cache_images:
        parser.error('--gpu-images requires --cache-images')
    if args.paired_manifest and (args.paper_mnist or args.gpu_images or args.cache_images):
        parser.error('Paired RGB inputs use their own fixed loader, not MNIST profiles/image caches')
    if args.paper_mnist:
        args.samples, args.vae_iter, args.mh_iter = 10000, 100, 100
        args.mutual_iterations, args.balanced, args.prior_scale = 5, True, 0.05
    if args.disable_mi:
        args.mutual_iterations = 1
    if args.mode not in (-1, 0, 1, 2) or args.prior_scale <= 0:
        parser.error("Modes: -1 MHNG, 0 local Gibbs, 1 all-accept, 2 central Gibbs; prior scale must be positive")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if min(args.mutual_iterations, args.samples, args.vae_iter, args.mh_iter, args.batch_size) <= 0:
        parser.error("Iteration counts, sample count and batch size must be positive")

    if args.debug:
        args.vae_iter = 2
        args.mh_iter = 2

    return args


def setup_directories(file_name=None) -> Dict[str, str]:
    """Create necessary directories for saving results."""
    model_dir = "./model"
    dir_name = str(file_name) if file_name else f"./model/run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    if os.path.exists(dir_name):
        raise FileExistsError(f"Refusing to overwrite an existing run: {dir_name}")

    directories = {
        "model_dir": model_dir,
        "dir_name": dir_name,
        "graphA_dir": f"{dir_name}/graphA",
        "graphB_dir": f"{dir_name}/graphB",
        "pth_dir": f"{dir_name}/pth",
        "npy_dir": f"{dir_name}/npy",
        "reconA_dir": f"{dir_name}/reconA",
        "reconB_dir": f"{dir_name}/reconB",
        "log_dir": f"{dir_name}/log",
        "result_dir": f"{dir_name}/result",
    }

    for dir_path in directories.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return directories


def prepare_datasets(batch_size: int, samples=10000, data_root="../data", balanced=False, cache_images=False) -> Tuple[Any, Any, Any, Any, int, np.ndarray]:
    """Prepare MNIST datasets for both agents with different rotation angles."""
    print("Dataset : MNIST")

    # Different rotation angles for each agent
    angle_a = 0  # MNIST's angle for Agent A
    angle_b = 45  # MNIST's angle for Agent B

    trans_ang1 = transforms.Compose(
        [transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.ToTensor()]
    )
    trans_ang2 = transforms.Compose(
        [transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.ToTensor()]
    )

    # Define datasets
    trainval_dataset1 = datasets.MNIST(
        data_root, train=True, transform=trans_ang1, download=True
    )  # Dataset for Agent A
    trainval_dataset2 = datasets.MNIST(
        data_root, train=True, transform=trans_ang2, download=True
    )  # Dataset for Agent B

    n_samples = len(trainval_dataset1)
    D = samples
    if not 0 < D <= n_samples:
        raise ValueError("Requested sample count is outside the MNIST training split")

    subset1_indices1 = list(range(0, D))
    if balanced:
        if D % 10:
            raise ValueError("Balanced MNIST requires sample count divisible by 10")
        labels = trainval_dataset1.targets.numpy()
        subset1_indices1 = sorted(np.concatenate([np.flatnonzero(labels == k)[:D // 10] for k in range(10)]).tolist())
    subset2_indices1 = list(range(D, n_samples))
    subset1_indices2 = list(range(0, D))
    subset2_indices2 = list(range(D, n_samples))

    train_dataset1 = Subset(trainval_dataset1, subset1_indices1)
    val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
    train_dataset2 = Subset(trainval_dataset2, subset1_indices1)
    val_dataset2 = Subset(trainval_dataset2, subset2_indices2)

    if cache_images:
        from runtime_cache import FixedRotationCache
        train_dataset1 = FixedRotationCache(train_dataset1, angle_a)
        train_dataset2 = FixedRotationCache(train_dataset2, angle_b)

    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=batch_size, shuffle=False
    )  # train_loader for agent A
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=batch_size, shuffle=False
    )  # train_loader for agent B
    all_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=batch_size, shuffle=False
    )
    all_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=batch_size, shuffle=False
    )

    return train_loader1, train_loader2, all_loader1, all_loader2, D, subset1_indices1


def train_vae_agents(
    args: argparse.Namespace,
    train_loader1,
    train_loader2,
    all_loader1,
    all_loader2,
    mu_d_A: np.ndarray,
    var_d_A: np.ndarray,
    mu_d_B: np.ndarray,
    var_d_B: np.ndarray,
    dir_name: str,
    iteration: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train VAE for both agents."""
    print(
        f"------------------Mutual learning session {iteration} begins------------------"
    )

    model_options = {}
    if getattr(args, 'paired_manifest', None):
        from vae_rgb64 import RGBVAE
        model_options['model_factory'] = RGBVAE

    # Train VAE module on Agent A
    c_nd_A, label, loss_list = cnn_vae_module_mnist.train(
        iteration=iteration,
        gmm_mu=torch.from_numpy(mu_d_A),
        gmm_var=torch.from_numpy(var_d_A),
        epoch=args.vae_iter,
        train_loader=train_loader1,
        batch_size=args.batch_size,
        all_loader=all_loader1,
        model_dir=dir_name,
        agent="A",
        cache_prior=getattr(args, 'cache_prior', False),
        gpu_images=getattr(args, 'gpu_images', False),
        epoch_sync=getattr(args, 'epoch_sync', False),
        **model_options,
    )

    # Train VAE module on Agent B
    c_nd_B, label, loss_list = cnn_vae_module_mnist.train(
        iteration=iteration,
        gmm_mu=torch.from_numpy(mu_d_B),
        gmm_var=torch.from_numpy(var_d_B),
        epoch=args.vae_iter,
        train_loader=train_loader2,
        batch_size=args.batch_size,
        all_loader=all_loader2,
        model_dir=dir_name,
        agent="B",
        cache_prior=getattr(args, 'cache_prior', False),
        gpu_images=getattr(args, 'gpu_images', False),
        epoch_sync=getattr(args, 'epoch_sync', False),
        **model_options,
    )

    return c_nd_A, c_nd_B, label


def initialize_gmm_parameters(K: int, dim: int, D: int, prior_scale=0.1) -> Dict[str, np.ndarray]:
    """Initialize GMM parameters for both agents."""
    # Set hyperparameters
    beta = 1.0
    m_d_A = np.repeat(0.0, dim)
    m_d_B = np.repeat(0.0, dim)
    w_dd_A = np.identity(dim) * prior_scale
    w_dd_B = np.identity(dim) * prior_scale
    nu = dim

    # Initialize μ, Λ
    mu_kd_A = np.empty((K, dim))
    lambda_kdd_A = np.empty((K, dim, dim))
    mu_kd_B = np.empty((K, dim))
    lambda_kdd_B = np.empty((K, dim, dim))

    for k in range(K):
        lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1)
        lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
        mu_kd_A[k] = np.random.multivariate_normal(
            mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])
        ).flatten()
        mu_kd_B[k] = np.random.multivariate_normal(
            mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])
        ).flatten()

    # Initialize unsampled w
    w_dk_A = np.random.multinomial(1, [1 / K] * K, size=D)
    w_dk_B = np.random.multinomial(1, [1 / K] * K, size=D)

    return {
        "beta": beta,
        "m_d_A": m_d_A,
        "m_d_B": m_d_B,
        "w_dd_A": w_dd_A,
        "w_dd_B": w_dd_B,
        "nu": nu,
        "mu_kd_A": mu_kd_A,
        "lambda_kdd_A": lambda_kdd_A,
        "mu_kd_B": mu_kd_B,
        "lambda_kdd_B": lambda_kdd_B,
        "w_dk_A": w_dk_A,
        "w_dk_B": w_dk_B,
    }


def compute_eta_matrix(
    c_nd: np.ndarray, mu_kd: np.ndarray, lambda_kdd: np.ndarray, K: int, D: int
) -> np.ndarray:
    """Compute eta matrix for probability calculations."""
    log_eta = np.empty((D, K))
    for k in range(K):
        sign, logdet = np.linalg.slogdet(lambda_kdd[k])
        if sign <= 0:
            raise ValueError("Precision matrix must have positive determinant")
        delta = c_nd - mu_kd[k]
        log_eta[:, k] = 0.5 * logdet - 0.5 * np.einsum(
            "di,ij,dj->d", delta, lambda_kdd[k], delta
        )
    return np.exp(log_eta - logsumexp(log_eta, axis=1, keepdims=True))


def metropolis_hastings_step_legacy(
    speaker_data: np.ndarray,
    listener_data: np.ndarray,
    speaker_mu: np.ndarray,
    speaker_lambda: np.ndarray,
    listener_mu: np.ndarray,
    listener_lambda: np.ndarray,
    w_dk_speaker: np.ndarray,
    w_dk_listener: np.ndarray,
    args: argparse.Namespace,
    K: int,
    D: int,
) -> Tuple[np.ndarray, int, list]:
    """Perform one step of Metropolis-Hastings algorithm."""
    # Algorithm 1 returns the listener's current word on rejection.
    # A proposal is temporary; it must not overwrite the speaker's stored word.
    w_dk = w_dk_listener.copy()
    count_accept = 0
    pred_labels = []

    # Compute eta matrix for speaker
    eta_dk_speaker = compute_eta_matrix(speaker_data, speaker_mu, speaker_lambda, K, D)

    cat_liks_speaker = np.zeros(D)
    cat_liks_listener = np.zeros(D)

    for d in range(D):
        # Sample w for speaker
        proposal = np.random.multinomial(
            n=1, pvals=eta_dk_speaker[d], size=1
        ).flatten()

        if args.mode == 0:  # No communication
            # Independent local Gibbs update (not an utterance).
            w_dk_speaker[d] = proposal
            pred_labels.append(np.argmax(w_dk_speaker[d]))
        elif args.mode == 1:  # All accept
            w_dk[d] = proposal
            count_accept += 1
            pred_labels.append(np.argmax(w_dk[d]))
        else:  # Normal Metropolis-Hastings
            cat_liks_speaker[d] = multivariate_normal.logpdf(
                listener_data[d],
                mean=listener_mu[np.argmax(proposal)],
                cov=np.linalg.inv(listener_lambda[np.argmax(proposal)]),
            )
            cat_liks_listener[d] = multivariate_normal.logpdf(
                listener_data[d],
                mean=listener_mu[np.argmax(w_dk_listener[d])],
                cov=np.linalg.inv(listener_lambda[np.argmax(w_dk_listener[d])]),
            )
            log_ratio = cat_liks_speaker[d] - cat_liks_listener[d]
            if not np.isfinite(log_ratio):
                raise FloatingPointError("Non-finite MH log likelihood ratio")
            rand_u = np.random.rand()

            if rand_u == 0.0 or np.log(rand_u) <= min(0.0, log_ratio):
                w_dk[d] = proposal
                count_accept += 1
            else:
                w_dk[d] = w_dk_listener[d]
            pred_labels.append(np.argmax(w_dk[d]))

    return w_dk, count_accept, pred_labels


def gaussian_log_likelihoods(data, means, precisions):
    """Unnormalized Gaussian log densities; common dimension constant cancels."""
    scores = np.empty((len(data), len(means)))
    for k, (mean, precision) in enumerate(zip(means, precisions)):
        np.linalg.cholesky(precision)  # Reject non-SPD, not just positive det.
        _, logdet = np.linalg.slogdet(precision)
        delta = data - mean
        scores[:, k] = 0.5 * logdet - 0.5 * np.einsum(
            'di,ij,dj->d', delta, precision, delta)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Non-finite Gaussian log likelihood')
    return scores


def metropolis_hastings_step(
    speaker_data, listener_data, speaker_mu, speaker_lambda,
    listener_mu, listener_lambda, w_dk_speaker, w_dk_listener, args, K, D,
):
    """Same sequential RNG calls/update semantics; density cache is direction-local."""
    eta = compute_eta_matrix(speaker_data, speaker_mu, speaker_lambda, K, D)
    listener_scores = (gaussian_log_likelihoods(listener_data, listener_mu, listener_lambda)
                       if args.mode not in (0, 1) else None)
    words = w_dk_listener.copy()
    accepted, labels = 0, []
    for d in range(D):
        proposal = np.random.multinomial(n=1, pvals=eta[d], size=1).flatten()
        if args.mode == 0:
            w_dk_speaker[d] = proposal
            labels.append(int(np.argmax(proposal)))
            continue
        if args.mode == 1:
            take = True
        else:
            ratio = (listener_scores[d, np.argmax(proposal)] -
                     listener_scores[d, np.argmax(w_dk_listener[d])])
            if not np.isfinite(ratio):
                raise FloatingPointError('Non-finite MH log likelihood ratio')
            u = np.random.rand()
            take = u == 0.0 or np.log(u) <= min(0.0, ratio)
        if take:
            words[d] = proposal
            accepted += 1
        labels.append(int(np.argmax(words[d])))
    return words, accepted, labels


def update_gmm_parameters(
    w_dk: np.ndarray,
    c_nd: np.ndarray,
    params: Dict[str, Any],
    agent_suffix: str,
    K: int,
    dim: int,
) -> None:
    """Update GMM parameters for one agent."""
    beta = params["beta"]
    m_d = params[f"m_d_{agent_suffix}"]
    w_dd = params[f"w_dd_{agent_suffix}"]
    nu = params["nu"]
    mu_kd = params[f"mu_kd_{agent_suffix}"]
    lambda_kdd = params[f"lambda_kdd_{agent_suffix}"]

    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, dim))
    w_hat_kdd = np.zeros((K, dim, dim))
    nu_hat_k = np.zeros(K)

    for k in range(K):
        beta_hat_k[k] = np.sum(w_dk[:, k]) + beta
        m_hat_kd[k] = np.sum(w_dk[:, k] * c_nd.T, axis=1)
        m_hat_kd[k] += beta * m_d
        m_hat_kd[k] /= beta_hat_k[k]

        tmp_w_dd = np.dot((w_dk[:, k] * c_nd.T), c_nd)
        tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
        tmp_w_dd -= beta_hat_k[k] * np.dot(
            m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim)
        )
        tmp_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        nu_hat_k[k] = np.sum(w_dk[:, k]) + nu

        # Sample λ and μ
        lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k],
            cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]),
            size=1,
        ).flatten()


def run_metropolis_hastings(
    args: argparse.Namespace,
    c_nd_A: np.ndarray,
    c_nd_B: np.ndarray,
    params: Dict[str, Any],
    K: int,
    D: int,
    dim: int,
    z_truth_n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete Metropolis-Hastings algorithm."""
    iteration = args.mh_iter
    ARI_A = np.zeros(iteration)
    ARI_B = np.zeros(iteration)
    concidence = np.zeros(iteration)
    accept_count_AtoB = np.zeros(iteration)
    accept_count_BtoA = np.zeros(iteration)

    print(f"M-H algorithm Start: Epoch:{iteration}")

    for i in range(iteration):
        pred_label_A = []
        pred_label_B = []

        # Speaker: A -> Listener: B
        w_dk, count_AtoB, pred_label_B_temp = metropolis_hastings_step(
            c_nd_A,
            c_nd_B,
            params["mu_kd_A"],
            params["lambda_kdd_A"],
            params["mu_kd_B"],
            params["lambda_kdd_B"],
            params["w_dk_A"],
            params["w_dk_B"],
            args,
            K,
            D,
        )

        if args.mode != 0:
            params["w_dk_B"][:] = w_dk
            pred_label_B.extend(pred_label_B_temp)

        # Update parameters for listener B
        if args.mode == -1 or args.mode == 1:
            update_gmm_parameters(w_dk, c_nd_B, params, "B", K, dim)

        if args.mode == 0:  # No communication mode - update speaker A
            update_gmm_parameters(params["w_dk_A"], c_nd_A, params, "A", K, dim)

        # Speaker: B -> Listener: A
        w_dk, count_BtoA, pred_label_A_temp = metropolis_hastings_step(
            c_nd_B,
            c_nd_A,
            params["mu_kd_B"],
            params["lambda_kdd_B"],
            params["mu_kd_A"],
            params["lambda_kdd_A"],
            params["w_dk_B"],
            params["w_dk_A"],
            args,
            K,
            D,
        )

        if args.mode != 0:
            params["w_dk_A"][:] = w_dk
            pred_label_A.extend(pred_label_A_temp)

        # Update parameters for listener A
        if args.mode == -1 or args.mode == 1:
            update_gmm_parameters(w_dk, c_nd_A, params, "A", K, dim)

        if args.mode == 0:  # No communication mode - update speaker B
            update_gmm_parameters(params["w_dk_B"], c_nd_B, params, "B", K, dim)

        # Fill pred_label_A and pred_label_B for mode 0
        if args.mode == 0:
            for d in range(D):
                pred_label_A.append(np.argmax(params["w_dk_A"][d]))
                pred_label_B.append(np.argmax(params["w_dk_B"][d]))

        # Evaluation
        # ARI is permutation-invariant; label matching is only for display.
        concidence[i] = cohen_kappa_score(pred_label_A, pred_label_B)
        ARI_A[i] = ari(z_truth_n, pred_label_A)
        ARI_B[i] = ari(z_truth_n, pred_label_B)
        accept_count_AtoB[i] = count_AtoB
        accept_count_BtoA[i] = count_BtoA

        if i == 0 or (i + 1) % 10 == 0 or i == (iteration - 1):
            print(
                f"=> Epoch: {i+1}, ARI_A: {ARI_A[i]}, ARI_B: {ARI_B[i]}, "
                f"Kappa:{concidence[i]}, A2B:{int(accept_count_AtoB[i])}, "
                f"B2A:{int(accept_count_BtoA[i])}"
            )

    return ARI_A, ARI_B, concidence, accept_count_AtoB, accept_count_BtoA


def central_word_probabilities(a, b, params):
    """Uniform-prior centralized word conditional, evaluated in log space."""
    scores = []
    K = len(params['mu_kd_A'])
    for agent, latent in [('A', a), ('B', b)]:
        scores.append(np.column_stack([
            multivariate_normal.logpdf(latent, mean=params[f'mu_kd_{agent}'][k],
                                      cov=np.linalg.inv(params[f'lambda_kdd_{agent}'][k]))
            for k in range(K)]))
    joint = scores[0] + scores[1]
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def run_central_gibbs(args, a, b, params, K, D, dim, labels):
    ari_a, ari_b = np.zeros(args.mh_iter), np.zeros(args.mh_iter)
    for iteration in range(args.mh_iter):
        probs = central_word_probabilities(a, b, params)
        words = np.array([np.random.multinomial(1, p) for p in probs])
        params['w_dk_A'][:] = words
        params['w_dk_B'][:] = words
        update_gmm_parameters(words, a, params, 'A', K, dim)
        update_gmm_parameters(words, b, params, 'B', K, dim)
        ari_a[iteration] = ari_b[iteration] = ari(labels, words.argmax(axis=1))
    # Common word agreement and acceptance counts are not communication metrics.
    return ari_a, ari_b, np.full(args.mh_iter, np.nan), np.zeros(args.mh_iter), np.zeros(args.mh_iter)


def feedback_priors(params, agent):
    """Local word -> own GMM mean and full covariance; no shared-label readout."""
    words = params[f"w_dk_{agent}"].argmax(axis=1)
    means = params[f"mu_kd_{agent}"][words].copy()
    covariance = np.linalg.inv(params[f"lambda_kdd_{agent}"])[words].copy()
    np.linalg.cholesky(covariance)  # Fail explicitly if the GMM prior is not SPD.
    return means, covariance


def save_results(
    params: Dict[str, Any],
    ARI_A: np.ndarray,
    ARI_B: np.ndarray,
    concidence: np.ndarray,
    directories: Dict[str, str],
    iteration: int,
) -> None:
    """Save model parameters and results."""
    npy_dir = directories["npy_dir"]
    log_dir = directories["log_dir"]

    np.save(f"{npy_dir}/muA_{iteration}.npy", params["mu_kd_A"])
    np.save(f"{npy_dir}/muB_{iteration}.npy", params["mu_kd_B"])
    np.save(f"{npy_dir}/lambdaA_{iteration}.npy", params["lambda_kdd_A"])
    np.save(f"{npy_dir}/lambdaB_{iteration}.npy", params["lambda_kdd_B"])
    np.savetxt(f"{log_dir}/ariA{iteration}.txt", ARI_A, fmt="%.3f")
    np.savetxt(f"{log_dir}/ariB{iteration}.txt", ARI_B, fmt="%.2f")
    np.savetxt(f"{log_dir}/cappa{iteration}.txt", concidence, fmt="%.2f")


def create_plots(
    ARI_A: np.ndarray,
    ARI_B: np.ndarray,
    concidence: np.ndarray,
    accept_count_AtoB: np.ndarray,
    accept_count_BtoA: np.ndarray,
    directories: Dict[str, str],
    iteration_num: int,
    total_iterations: int,
    D: int,
) -> None:
    """Create evaluation plots."""
    result_dir = directories["result_dir"]

    # Acceptance plot
    plt.figure()
    plt.plot(
        range(0, total_iterations),
        accept_count_AtoB,
        marker="None",
        label="Accept_num:AtoB",
    )
    plt.plot(
        range(0, total_iterations),
        accept_count_BtoA,
        marker="None",
        label="Accept_num:BtoA",
    )
    plt.xlabel("iteration")
    plt.ylabel("Number of acceptation")
    plt.ylim(0, D)
    plt.legend()
    plt.savefig(f"{result_dir}/accept{iteration_num}.png")
    plt.close()

    # Concidence plot
    plt.figure()
    plt.plot(range(0, total_iterations), concidence, marker="None")
    plt.xlabel("iteration")
    plt.ylabel("Concidence")
    plt.ylim(0, 1)
    plt.title("k")
    plt.savefig(f"{result_dir}/conf{iteration_num}.png")
    plt.close()

    # ARI plot
    plt.figure()
    plt.plot(range(0, total_iterations), ARI_A, marker="None", label="ARI_A")
    plt.plot(range(0, total_iterations), ARI_B, marker="None", label="ARI_B")
    plt.xlabel("iteration")
    plt.ylabel("ARI")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("ARI")
    plt.savefig(f"{result_dir}/ari{iteration_num}.png")
    plt.close()


def main():
    """Main function orchestrating the entire training process."""
    # Parse arguments and setup
    args = parse_arguments()
    device = torch.device("cuda" if args.cuda else "cpu")
    cnn_vae_module_mnist.device = device
    print("CUDA", args.cuda)

    # Setup directories
    directories = setup_directories(args.run_dir)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('main.py', 'cnn_vae_module_mnist.py')}
    (Path(directories['dir_name']) / 'run.json').write_text(
        json.dumps({'args': vars(args), 'source_sha256': source_hashes,
                    'torch': torch.__version__, 'numpy': np.__version__}, indent=2), encoding='utf-8')

    # Prepare datasets
    if args.paired_manifest:
        from paired_images import prepare_paired
        train_loader1, train_loader2, all_loader1, all_loader2, D, indices = prepare_paired(args.paired_manifest, args.batch_size)
        args.samples = D
        metadata = Path(args.paired_manifest).read_bytes()
        (Path(directories['dir_name'])/'paired-manifest.json').write_bytes(metadata)
        run_record = json.loads((Path(directories['dir_name'])/'run.json').read_text())
        run_record['args']['samples'] = D
        run_record['paired_manifest_sha256'] = hashlib.sha256(metadata).hexdigest()
        for filename in ('paired_images.py','vae_rgb64.py'):
            run_record['source_sha256'][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        (Path(directories['dir_name'])/'run.json').write_text(json.dumps(run_record,indent=2))
    else:
        train_loader1, train_loader2, all_loader1, all_loader2, D, indices = prepare_datasets(
            args.batch_size, args.samples, args.data_root, args.balanced, args.cache_images
        )
    np.save(Path(directories['npy_dir']) / 'image_indices.npy', indices)

    print(f"Total data:{D}, Category:{args.category}")
    print(f"VAE_iter:{args.vae_iter}, Batch_size:{args.batch_size}")
    print(f"MH_iter:{args.mh_iter}, MH_mode:{args.mode}(-1:Com 0:No-com 1:All accept)")

    # Initialize variables for mutual learning
    mutual_iteration = args.mutual_iterations
    mu_d_A = np.zeros(D)
    var_d_A = np.zeros(D)
    mu_d_B = np.zeros(D)
    var_d_B = np.zeros(D)
    params = None

    for it in range(mutual_iteration):
        # Train VAE agents
        c_nd_A, c_nd_B, label = train_vae_agents(
            args,
            train_loader1,
            train_loader2,
            all_loader1,
            all_loader2,
            mu_d_A,
            var_d_A,
            mu_d_B,
            var_d_B,
            directories["dir_name"],
            it,
        )

        K = args.category
        z_truth_n = label
        dim = len(c_nd_A[0])

        # Initialize GMM parameters
        if params is None:
            params = initialize_gmm_parameters(K, dim, D, args.prior_scale)

        # Run Metropolis-Hastings algorithm
        (
            ARI_A,
            ARI_B,
            concidence,
            accept_count_AtoB,
            accept_count_BtoA,
        ) = (run_central_gibbs if args.mode == 2 else run_metropolis_hastings)(
            args, c_nd_A, c_nd_B, params, K, D, dim, z_truth_n)

        # Save results
        save_results(params, ARI_A, ARI_B, concidence, directories, it)
        np.savez(Path(directories['npy_dir']) / f'metrics_{it}.npz',
                 ari_a=ARI_A, ari_b=ARI_B, kappa=concidence,
                 acceptance_a_to_b=accept_count_AtoB, acceptance_b_to_a=accept_count_BtoA)
        mu_d_A, var_d_A = feedback_priors(params, "A")
        mu_d_B, var_d_B = feedback_priors(params, "B")
        for agent, mean, covariance in [('A', mu_d_A, var_d_A), ('B', mu_d_B, var_d_B)]:
            np.savez(Path(directories['npy_dir']) / f'feedback{agent}_{it}.npz',
                     mean=mean, covariance=covariance, words=params[f'w_dk_{agent}'])

        # Create plots
        create_plots(
            ARI_A,
            ARI_B,
            concidence,
            accept_count_AtoB,
            accept_count_BtoA,
            directories,
            it,
            args.mh_iter,
            D,
        )

        # Create confusion matrices
        _, result_a = calc_ari(
            [np.argmax(params["w_dk_A"][d]) for d in range(D)], z_truth_n
        )
        _, result_b = calc_ari(
            [np.argmax(params["w_dk_B"][d]) for d in range(D)], z_truth_n
        )
        cmx(
            iteration=it,
            y_true=z_truth_n,
            y_pred=result_a,
            agent="A",
            save_dir=directories["result_dir"],
        )
        cmx(
            iteration=it,
            y_true=z_truth_n,
            y_pred=result_b,
            agent="B",
            save_dir=directories["result_dir"],
        )

        print(
            f"Iteration:{it} Done:max_ARI_A: {max(ARI_A)}, max_ARI_B: {max(ARI_B)}, max_Kappa:{max(concidence)}"
        )


if __name__ == "__main__":
    main()
