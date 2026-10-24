"""Optional caches: preserve fixed-image RNG draws and full Gaussian KL."""
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation


class FixedRotationCache(Dataset):
    def __init__(self, dataset, angle):
        self.angle = angle
        # Cache construction must not consume training RNG state.
        with torch.random.fork_rng(devices=[]):
            pairs = [dataset[i] for i in range(len(dataset))]
        self.images = torch.stack([x for x, _ in pairs])
        self.labels = torch.tensor([y for _, y in pairs])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Original fixed RandomRotation still draws an angle on every access.
        RandomRotation.get_params([self.angle, self.angle])
        return self.images[index], self.labels[index]


def prepare_gaussian_prior(mean, covariance, device):
    mean = torch.as_tensor(mean, dtype=torch.float64, device=device)
    covariance = torch.as_tensor(covariance, dtype=torch.float64, device=device)
    chol = torch.linalg.cholesky(covariance)
    return mean, torch.cholesky_inverse(chol), 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def cached_gaussian_kl(mu, logvar, prior):
    mu, logvar = mu.double(), logvar.double()
    mean, precision, logdet = prior
    delta = mu - mean
    return .5 * ((precision.diagonal(dim1=-2, dim2=-1) * logvar.exp()).sum(-1)
        + torch.einsum('bi,bij,bj->b', delta, precision, delta)
        - mu.shape[-1] + logdet - logvar.sum(-1))


class GPUResidentBatches:
    """Sequential single-worker loader, preserving its CPU RNG consumption."""
    def __init__(self, loader, device):
        from torch.utils.data import SequentialSampler
        if (not isinstance(loader.dataset, FixedRotationCache) or
                not isinstance(loader.sampler, SequentialSampler) or
                loader.num_workers != 0 or loader.drop_last):
            raise ValueError('GPU residency requires fixed cache, sequential complete single-worker loader')
        self.loader = loader
        self.images = loader.dataset.images.to(device)

    def __iter__(self):
        # Constructing the original iterator consumes its exact base-seed draw.
        iterator = iter(self.loader)
        del iterator
        n, batch = len(self.loader.dataset), self.loader.batch_size
        for start in range(0, n, batch):
            end = min(start + batch, n)
            for _ in range(start, end):
                RandomRotation.get_params([self.loader.dataset.angle] * 2)
            yield self.images[start:end], self.loader.dataset.labels[start:end]


def finish_epoch_losses(losses):
    """One host transfer; retain original ordered Python-float summation."""
    values = torch.stack(losses).cpu().tolist()
    import math
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError('Non-finite VAE objective (detected at epoch boundary; discard run)')
    total = 0
    for value in values:
        total += value
    return total
