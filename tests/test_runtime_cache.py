import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from runtime_cache import FixedRotationCache, prepare_gaussian_prior, cached_gaussian_kl
from cnn_vae_module_mnist import gaussian_kl


class Images(Dataset):
    def __init__(self, angle):
        self.transform = transforms.Compose([transforms.RandomRotation((angle, angle)), transforms.ToTensor()])
    def __len__(self):
        return 17
    def __getitem__(self, i):
        image = Image.fromarray((np.arange(784).reshape(28, 28) + i).astype('uint8'))
        return self.transform(image), i % 10


def test_images_and_rng_exact():
    for angle in (0, 45):
        data = Images(angle)
        torch.manual_seed(12)
        before = torch.get_rng_state()
        cache = FixedRotationCache(data, angle)
        assert torch.equal(before, torch.get_rng_state())
        for _ in range(2):
            before = torch.get_rng_state()
            old = [data[i] for i in range(len(data))]
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            new = [cache[i] for i in range(len(cache))]
            assert torch.equal(after, torch.get_rng_state())
            for (x, y), (xx, yy) in zip(old, new):
                assert torch.equal(x, xx) and y == yy


def test_kl_values_and_gradients():
    torch.manual_seed(17)
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        mu = torch.randn(19, 12, device=device, requires_grad=True)
        lv = torch.randn(19, 12, device=device, requires_grad=True)
        mean = torch.randn(19, 12)
        x = torch.randn(19, 12, 12, dtype=torch.float64)
        cov = x @ x.transpose(-1, -2) + torch.eye(12)
        old = gaussian_kl(mu, lv, mean, cov)
        grad = torch.autograd.grad(old.sum(), (mu, lv))
        new = cached_gaussian_kl(mu, lv, prepare_gaussian_prior(mean, cov, device))
        new_grad = torch.autograd.grad(new.sum(), (mu, lv))
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        for a, b in zip(grad, new_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
