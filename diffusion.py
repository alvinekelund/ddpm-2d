import torch
import numpy as np


class Diffusion:
    """
    DDPM noise schedule and forward/backward process.
    Linear beta schedule from beta_start to beta_end over T steps.
    """
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()

    def q_sample(self, x0, t):
        """Forward process: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps"""
        eps = torch.randn_like(x0)
        s_ab = self.sqrt_alpha_bars.to(x0.device)[t].view(-1, 1)
        s_1m = self.sqrt_one_minus_alpha_bars.to(x0.device)[t].view(-1, 1)
        return s_ab * x0 + s_1m * eps, eps

    def p_sample(self, model, x_t, t_int):
        """One reverse step: x_{t-1} from x_t using the model."""
        model.eval()
        with torch.no_grad():
            t_batch = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
            eps_pred = model(x_t, t_batch)

        d = x_t.device
        alpha = self.alphas[t_int].to(d)
        alpha_bar = self.alpha_bars[t_int].to(d)
        beta = self.betas[t_int].to(d)

        coef = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
        mean = self.sqrt_recip_alphas[t_int].to(d) * (x_t - coef * eps_pred)

        if t_int > 0:
            return mean + beta.sqrt() * torch.randn_like(x_t)
        return mean

    def sample(self, model, n, device, save_every=50):
        """Full reverse process from noise to data. Returns samples and trajectory."""
        x = torch.randn(n, 2, device=device)
        trajectory = {self.T: x.cpu().numpy()}

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t % save_every == 0:
                trajectory[t] = x.cpu().numpy()

        return x.cpu().numpy(), trajectory
