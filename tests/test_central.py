import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def test_joint_word_conditional_matches_direct_likelihood_product():
    params = {'mu_kd_A': np.array([[0.], [2.]]), 'mu_kd_B': np.array([[-1.], [1.]]),
              'lambda_kdd_A': np.ones((2, 1, 1)), 'lambda_kdd_B': np.ones((2, 1, 1))}
    a, b = np.array([[.2], [.7]]), np.array([[.4], [-.5]])
    direct = np.column_stack([norm.pdf(a[:, 0], params['mu_kd_A'][k, 0]) * norm.pdf(b[:, 0], params['mu_kd_B'][k, 0]) for k in range(2)])
    direct /= direct.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(main.central_word_probabilities(a, b, params), direct, rtol=1e-13, atol=1e-15)


def test_balanced_indices_are_common_and_balanced(monkeypatch):
    import torch
    class FakeMNIST:
        targets = torch.tensor(list(range(10)) * 3)
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return len(self.targets)
    monkeypatch.setattr(main.datasets, 'MNIST', FakeMNIST)
    a, b, _, _, count, indices = main.prepare_datasets(5, 20, 'unused', True)
    assert count == 20 and a.dataset.indices == b.dataset.indices == indices
    np.testing.assert_array_equal(np.bincount(FakeMNIST.targets[indices].numpy()), np.full(10, 2))
