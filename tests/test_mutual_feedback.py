"""Numerical and integration tests without dataset downloads."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as naming
import cnn_vae_module_mnist as vae


def test_full_covariance_kl_matches_torch_and_has_gradients():
    mu = torch.tensor([[.2, -.7], [1., .5]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[.1, -.2], [-.5, .4]], dtype=torch.float64, requires_grad=True)
    pm = torch.zeros_like(mu)
    cov = torch.tensor([[[2., .8], [.8, 1.]], [[1., -.3], [-.3, 2.]]], dtype=torch.float64)
    got = vae.gaussian_kl(mu, logvar, pm, cov)
    ref = torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(mu, covariance_matrix=torch.diag_embed(logvar.exp())),
        torch.distributions.MultivariateNormal(pm, covariance_matrix=cov))
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(got, vae.gaussian_kl(mu, logvar, pm, torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1))))
    got.sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(logvar.grad).all()


def test_standard_normal_kl_matches_initial_objective():
    mu = torch.tensor([[.2, -.1]], dtype=torch.float64)
    lv = torch.tensor([[.4, -.5]], dtype=torch.float64)
    got = vae.gaussian_kl(mu, lv, torch.zeros_like(mu), torch.eye(2).unsqueeze(0))
    torch.testing.assert_close(got, -.5 * (1 + lv - mu.square() - lv.exp()).sum(-1), rtol=1e-12, atol=1e-12)


def test_feedback_uses_each_agents_private_words_and_full_covariance():
    covs = np.array([[[2., .5], [.5, 1.]], [[1., -.2], [-.2, 3.]]])
    params = {}
    for a, words, offset in [('A', [0, 1], 0), ('B', [1, 0], 10)]:
        params[f'w_dk_{a}'] = np.eye(2)[words]
        params[f'mu_kd_{a}'] = np.array([[0., 1.], [2., 3.]]) + offset
        params[f'lambda_kdd_{a}'] = np.linalg.inv(covs)
        means, covariance = naming.feedback_priors(params, a)
        np.testing.assert_array_equal(means, params[f'mu_kd_{a}'][words])
        np.testing.assert_allclose(covariance, covs[words], rtol=1e-13, atol=1e-15)


class TinyVAE(torch.nn.Module):
    loss_function = vae.VAE.loss_function
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 24)
        self.decoder = torch.nn.Linear(12, 4)
    def encode(self, x):
        mu, lv = self.encoder(x).chunk(2, -1)
        return mu + (lv / 2).exp() * torch.randn_like(mu), mu, lv
    def forward(self, x):
        z, mu, lv = self.encode(x)
        return self.decoder(z).sigmoid(), mu, lv, z


def test_two_round_training_resumes_optimizer_and_consumes_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(vae, 'VAE', TinyVAE)
    monkeypatch.setattr(vae, 'device', torch.device('cpu'))
    for folder in ['npy', 'pth', 'log', 'graphA']:
        (tmp_path / folder).mkdir()
    ds = TensorDataset(torch.rand(5, 4), torch.arange(5))
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    z0, labels0, _ = vae.train(0, None, None, 1, loader, 2, loader, str(tmp_path), 'A')
    prior_mu = torch.ones(5, 12)
    prior_cov = torch.eye(12).repeat(5, 1, 1)
    prior_cov[:, 0, 1] = prior_cov[:, 1, 0] = .2
    z1, labels1, _ = vae.train(1, prior_mu, prior_cov, 1, loader, 2, loader, str(tmp_path), 'A')
    assert z0.shape == z1.shape == (5, 12)
    np.testing.assert_array_equal(labels0, np.arange(5))
    np.testing.assert_array_equal(labels1, np.arange(5))
    state0 = torch.load(tmp_path / 'pth/trainA_0.pth', weights_only=True)
    state1 = torch.load(tmp_path / 'pth/trainA_1.pth', weights_only=True)
    assert next(iter(state0['optimizer']['state'].values()))['step'].item() == 3
    assert next(iter(state1['optimizer']['state'].values()))['step'].item() == 6
    saved = np.load(tmp_path / 'npy/used_priorA_1.npz')
    np.testing.assert_array_equal(saved['covariance'], prior_cov.numpy())
    assert not torch.equal(state0['model']['encoder.weight'], state1['model']['encoder.weight'])


def test_shuffled_prior_order_is_rejected(tmp_path):
    ds = TensorDataset(torch.rand(5, 4), torch.arange(5))
    shuffled = DataLoader(ds, batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match='sequential'):
        vae.train(0, None, None, 1, shuffled, 2, shuffled, str(tmp_path), 'A')
