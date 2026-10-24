import pytest
import torch
from torch.utils.data import DataLoader
from runtime_cache import GPUResidentBatches, FixedRotationCache, finish_epoch_losses
from test_runtime_cache import Images


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_resident_pixels_order_and_rng():
    loader = DataLoader(FixedRotationCache(Images(45), 45), batch_size=8, shuffle=False)
    start = torch.get_rng_state()
    resident = GPUResidentBatches(loader, 'cuda')
    assert torch.equal(start, torch.get_rng_state())
    for _ in range(2):
        start = torch.get_rng_state()
        original = list(loader)
        end = torch.get_rng_state()
        torch.set_rng_state(start)
        optimized = list(resident)
        assert torch.equal(end, torch.get_rng_state())
        for (x, y), (xx, yy) in zip(original, optimized):
            assert torch.equal(x, xx.cpu()) and torch.equal(y, yy)


def test_epoch_sum_and_nonfinite_failure():
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        values = [torch.tensor(x, device=device) for x in (1e8, .1, 2., -.2)]
        # Oracle is the actual legacy += loop, not Python 3.12's compensated sum.
        legacy_total = 0
        for value in values:
            legacy_total += value.item()
        assert finish_epoch_losses(values) == legacy_total
        with pytest.raises(FloatingPointError):
            finish_epoch_losses([torch.tensor(float('nan'), device=device)])
