import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from model import ScoreNet
from diffusion import Diffusion
from data import get


def load(dataset, T=500):
    ckpt = Path(f"checkpoints_{dataset}/model.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} — run train.py {dataset} first")
    model = ScoreNet(T=T)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model


def plot_forward_process(dataset="moons", out="forward_process.png"):
    """Show what happens to data as noise is progressively added."""
    diffusion = Diffusion()
    data = torch.tensor(get(dataset, n=2000))
    timesteps = [0, 50, 100, 200, 350, 499]

    fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 2.5))
    for ax, t_int in zip(axes, timesteps):
        t = torch.full((len(data),), t_int, dtype=torch.long)
        x_t, _ = diffusion.q_sample(data, t)
        x = x_t.numpy()
        ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.4, c="steelblue", rasterized=True)
        ax.set_title(f"t = {t_int}")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis("off")

    fig.suptitle(f"Forward process: adding noise to {dataset}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_reverse_process(dataset, out=None):
    """Show the denoising trajectory from noise to data."""
    out = out or f"reverse_process_{dataset}.png"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    diffusion = Diffusion()
    model = load(dataset).to(device)

    _, traj = diffusion.sample(model, n=2000, device=device, save_every=50)
    steps_shown = sorted(traj.keys(), reverse=True)[:6]

    fig, axes = plt.subplots(1, len(steps_shown), figsize=(14, 2.5))
    for ax, t_int in zip(axes, steps_shown):
        x = traj[t_int]
        ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.4, c="coral", rasterized=True)
        ax.set_title(f"t = {t_int}")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis("off")

    fig.suptitle(f"Reverse process: denoising back to {dataset}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_samples_vs_real(datasets=("moons", "checkerboard", "rings"), out="samples_vs_real.png"):
    """Side-by-side: real data vs model samples."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    diffusion = Diffusion()
    n = 2000

    fig, axes = plt.subplots(2, len(datasets), figsize=(12, 5))
    for col, name in enumerate(datasets):
        try:
            model = load(name).to(device)
        except FileNotFoundError:
            print(f"skipping {name} — not trained yet")
            continue

        real = get(name, n=n)
        samples, _ = diffusion.sample(model, n=n, device=device)

        for row, (x, title, color) in enumerate([
            (real, f"{name}\nreal data", "steelblue"),
            (samples, f"{name}\ngenerated", "coral"),
        ]):
            ax = axes[row][col]
            ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.4, c=color, rasterized=True)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis("off")
            ax.set_title(title, fontsize=9)

    fig.suptitle("Real data vs. diffusion model samples")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_score_field(dataset, t_int=100, out=None, n_grid=20):
    """Visualise the learned score function (noise prediction) as an arrow field."""
    out = out or f"score_field_{dataset}.png"
    model = load(dataset)
    model.eval()

    lim = 3.5
    xs = np.linspace(-lim, lim, n_grid)
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    x_grid = torch.tensor(grid, dtype=torch.float)
    t_batch = torch.full((len(x_grid),), t_int, dtype=torch.long)

    with torch.no_grad():
        score = model(x_grid, t_batch).numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    real = get(dataset, n=1000)
    ax.scatter(real[:, 0], real[:, 1], s=2, alpha=0.2, c="steelblue", zorder=0, rasterized=True)
    ax.quiver(
        grid[:, 0], grid[:, 1],
        -score[:, 0], -score[:, 1],
        alpha=0.7, color="coral", scale=40,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    ax.set_title(f"Learned score field at t={t_int}\n(arrows point toward data)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    datasets_to_run = [dataset] if dataset else ["moons", "checkerboard", "rings"]
    trained = [d for d in datasets_to_run if Path(f"checkpoints_{d}/model.pt").exists()]

    if not trained:
        print("no trained models found — run train.py first")
    else:
        plot_forward_process(trained[0])
        for d in trained:
            plot_reverse_process(d)
            plot_score_field(d)
        plot_samples_vs_real(tuple(trained))
