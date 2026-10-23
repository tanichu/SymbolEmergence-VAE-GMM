"""Verify saved full-prior transfer and optimizer continuation for a smoke run."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def check(root):
    root = Path(root)
    config = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    rounds = config['args']['mutual_iterations']
    samples = config['args']['samples']
    steps_per_round = int(np.ceil(samples / config['args']['batch_size'])) * config['args']['vae_iter']
    report = {'rounds': rounds, 'samples': samples, 'checks': []}
    for agent in ['A', 'B']:
        for iteration in range(rounds):
            state = torch.load(root / 'pth' / f'train{agent}_{iteration}.pth', map_location='cpu', weights_only=True)
            steps = [int(v['step']) for v in state['optimizer']['state'].values()]
            assert steps and set(steps) == {steps_per_round * (iteration + 1)}
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            meta = json.loads((root / 'log' / f'vae{agent}_{iteration}.json').read_text(encoding='utf-8'))
            assert np.isfinite(meta['final_negative_objective_per_image'])
            if iteration:
                assert meta['prior'] == 'full-gmm' and meta['resumed_from']
                with np.load(root / 'npy' / f'feedback{agent}_{iteration-1}.npz') as feedback, np.load(root / 'npy' / f'used_prior{agent}_{iteration}.npz') as used:
                    np.testing.assert_array_equal(used['mean'], feedback['mean'])
                    np.testing.assert_array_equal(used['covariance'], feedback['covariance'])
                    assert used['mean'].shape == (samples, 12)
                    assert used['covariance'].shape == (samples, 12, 12)
                    np.linalg.cholesky(used['covariance'])
                    offdiag = used['covariance'].copy()
                    offdiag[:, np.arange(12), np.arange(12)] = 0
                    assert np.any(offdiag != 0)
                    report['checks'].append({'agent': agent, 'round': iteration,
                        'prior_transfer': 'bitwise equal', 'optimizer_steps': steps[0],
                        'max_abs_offdiagonal': float(np.max(np.abs(offdiag)))})
    report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    report = check(args.run_dir)
    (Path(args.run_dir) / 'feedback_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
