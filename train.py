import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from model import ScoreNet
from diffusion import Diffusion
from data import get


def run(
    dataset="moons",
    T=500,
    d_hidden=256,
    n_layers=4,
    steps=15000,
    batch_size=1024,
    lr=3e-4,
    log_every=500,
    save_dir=None,
    seed=0,
):
    torch.manual_seed(seed)
    device = "cpu"
    print(f"dataset: {dataset}  device: {device}")

    if save_dir is None:
        save_dir = f"checkpoints_{dataset}"
    Path(save_dir).mkdir(exist_ok=True)

    data = torch.tensor(get(dataset, n=50000))
    diffusion = Diffusion(T=T)
    model = ScoreNet(d_hidden=d_hidden, n_layers=n_layers, T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    for step in range(steps + 1):
        model.train()
        idx = torch.randint(len(data), (batch_size,))
        x0 = data[idx].to(device)
        t = torch.randint(0, T, (batch_size,), device=device)
        x_t, eps = diffusion.q_sample(x0, t)
        loss = F.mse_loss(model(x_t, t), eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0:
            log.append({"step": step, "loss": round(loss.item(), 5)})
            with open(f"{save_dir}/log.json", "w") as f:
                json.dump(log, f)
            print(f"step {step:6d}  loss {loss.item():.5f}")

    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    print("done")
    return model, diffusion


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "moons"
    run(dataset=dataset)
