"""Synthetic regressions: no image download or training is required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("reproduction_main", ROOT / "main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def exchange(monkeypatch, mode, uniform, listener_x=0.0):
    # Deterministic proposal to category 1, current listener category 0.
    monkeypatch.setattr(np.random, "multinomial", lambda **kw: np.array([[0, 1]]))
    monkeypatch.setattr(np.random, "rand", lambda: uniform)
    speaker = np.array([[1, 0]])
    listener = np.array([[1, 0]])
    means = np.array([[0.0], [4.0]])
    precision = np.ones((2, 1, 1))
    result, count, _ = module.metropolis_hastings_step(
        np.zeros((1, 1)), np.array([[listener_x]]), means, precision,
        means, precision, speaker, listener, SimpleNamespace(mode=mode), 2, 1,
    )
    return result, count, speaker, listener


@pytest.mark.parametrize("mode,uniform,expected", [(-1, 0.5, 0), (-1, 0.0001, 1), (1, 0.5, 1)])
def test_proposal_does_not_mutate_stored_arrays(monkeypatch, mode, uniform, expected):
    result, count, speaker, listener = exchange(monkeypatch, mode, uniform)
    assert result.argmax() == expected
    assert count == expected
    np.testing.assert_array_equal(speaker, [[1, 0]])
    np.testing.assert_array_equal(listener, [[1, 0]])


def test_no_comm_updates_own_word_only(monkeypatch):
    _, count, speaker, listener = exchange(monkeypatch, 0, 0.5)
    np.testing.assert_array_equal(speaker, [[0, 1]])
    np.testing.assert_array_equal(listener, [[1, 0]])
    assert count == 0


def test_equal_extreme_densities_accept_without_underflow(monkeypatch):
    # Both ordinary Gaussian PDFs underflow, but their log ratio is positive.
    result, count, _, _ = exchange(monkeypatch, -1, 0.5, listener_x=1000.0)
    assert result.argmax() == 1
    assert count == 1


def test_proposal_probabilities_match_direct_gaussians():
    x = np.array([[-1., 2.], [0., 1.]])
    means = np.array([[0., 0.], [1., 1.]])
    precisions = np.array([[[2., .2], [.2, 1.]], [[1., 0.], [0., 3.]]])
    ref = np.column_stack([multivariate_normal.pdf(x, mean=means[k], cov=np.linalg.inv(precisions[k])) for k in range(2)])
    ref /= ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(module.compute_eta_matrix(x, means, precisions, 2, 2), ref, rtol=1e-13, atol=1e-15)


def test_sweep_assigns_listener_result_before_parameter_update(monkeypatch):
    params = {"w_dk_A": np.array([[1, 0]]), "w_dk_B": np.array([[1, 0]]),
              "mu_kd_A": np.zeros((2, 1)), "mu_kd_B": np.zeros((2, 1)),
              "lambda_kdd_A": np.ones((2, 1, 1)), "lambda_kdd_B": np.ones((2, 1, 1))}
    seen = []
    def step(*args):
        return np.array([[0, 1]]), 1, [1]
    def update(words, data, p, agent, K, dim):
        np.testing.assert_array_equal(p[f"w_dk_{agent}"], words)
        seen.append(agent)
    monkeypatch.setattr(module, "metropolis_hastings_step", step)
    monkeypatch.setattr(module, "update_gmm_parameters", update)
    monkeypatch.setattr(module, "calc_ari", lambda pred, truth: (None, pred))
    module.run_metropolis_hastings(SimpleNamespace(mh_iter=1, mode=-1),
        np.zeros((1, 1)), np.zeros((1, 1)), params, 2, 1, 1, np.array([0]))
    assert seen == ["B", "A"]
    np.testing.assert_array_equal(params["w_dk_A"], [[0, 1]])
    np.testing.assert_array_equal(params["w_dk_B"], [[0, 1]])
