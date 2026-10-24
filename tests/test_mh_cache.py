import copy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.stats import multivariate_normal
import main as m


def fixture(seed=19, n=128):
    rng = np.random.default_rng(seed)
    k, p = 10, 12
    data = [rng.normal(size=(n, p)) for _ in range(2)]
    means = [rng.normal(size=(k, p)) for _ in range(2)]
    precisions = []
    for _ in range(2):
        x = rng.normal(size=(k, p, p))
        precisions.append(x @ x.transpose(0, 2, 1) + np.eye(p))
    words = [np.eye(k, dtype=int)[rng.integers(k, size=n)] for _ in range(2)]
    return data[0], data[1], means[0], precisions[0], means[1], precisions[1], *words


def test_density_ratios():
    a = fixture()
    scores = m.gaussian_log_likelihoods(a[1], a[4], a[5])
    reference = np.column_stack([multivariate_normal.logpdf(
        a[1], mean=a[4][k], cov=np.linalg.inv(a[5][k])) for k in range(10)])
    np.testing.assert_allclose(scores - scores[:, :1], reference - reference[:, :1],
                               rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize('mode', [-1, 0, 1])
@pytest.mark.parametrize('seed', [0, 4, 15])
def test_step_and_rng(mode, seed):
    args = (*fixture(seed), SimpleNamespace(mode=mode), 10, 128)
    outcomes = []
    for kernel in (m.metropolis_hastings_step_legacy, m.metropolis_hastings_step):
        inp = copy.deepcopy(args)
        np.random.seed(seed + 100)
        result = kernel(*inp)
        outcomes.append((result, inp[6:8], np.random.get_state()))
    for i in (0, 1):
        for old, new in zip(outcomes[0][i], outcomes[1][i]):
            np.testing.assert_array_equal(old, new)
    assert outcomes[0][2][0] == outcomes[1][2][0]
    np.testing.assert_array_equal(outcomes[0][2][1], outcomes[1][2][1])
    assert outcomes[0][2][2:] == outcomes[1][2][2:]


def test_cache_rebuilt_after_parameter_change():
    a = fixture()
    before = m.gaussian_log_likelihoods(a[1], a[4], a[5])
    after = m.gaussian_log_likelihoods(a[1], a[4] + 1, a[5])
    assert not np.array_equal(before, after)


def test_non_spd_rejected():
    a = fixture()
    a[5][0] = -np.eye(12)
    with pytest.raises(np.linalg.LinAlgError):
        m.gaussian_log_likelihoods(a[1], a[4], a[5])
