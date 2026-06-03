import numpy as np


def make_moons(n=10000, noise=0.05):
    n_each = n // 2
    t = np.linspace(0, np.pi, n_each)
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    x = np.concatenate([x1, x2]) + np.random.randn(n, 2) * noise
    return (x - x.mean(0)) / x.std()


def make_checkerboard(n=10000, noise=0.02):
    collected = []
    total = 0
    while total < n:
        xy = np.random.rand(n * 2, 2) * 4 - 2
        mask = (np.floor(xy[:, 0]) + np.floor(xy[:, 1])) % 2 == 0
        collected.append(xy[mask])
        total += mask.sum()
    x = np.concatenate(collected)[:n] + np.random.randn(n, 2) * noise
    return (x - x.mean(0)) / x.std()


def make_rings(n=10000, noise=0.05):
    n_each = n // 2
    t = np.random.rand(n_each) * 2 * np.pi
    r1 = 1.0 + np.random.randn(n_each) * noise
    r2 = 2.2 + np.random.randn(n_each) * noise
    x1 = np.stack([r1 * np.cos(t), r1 * np.sin(t)], axis=1)
    x2 = np.stack([r2 * np.cos(t), r2 * np.sin(t)], axis=1)
    x = np.concatenate([x1, x2])
    return x / x.std()


DATASETS = {
    "moons": make_moons,
    "checkerboard": make_checkerboard,
    "rings": make_rings,
}


def get(name, n=50000):
    return DATASETS[name](n).astype(np.float32)
