import os
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from matsciml.datasets import registry as ds_reg
from monai.generative.schedulers import DDPMScheduler

# --------------------------
# Data: 9D lattice vectors with z-score normalization
# --------------------------
class LatticeVecs(Dataset):
    def __init__(self, base):
        self.base = base
        lat = [self._l9(i) for i in range(len(base))]
        X = torch.stack(lat)  # (N, 9)
        self.mean = X.mean(0)
        self.std = X.std(0).clamp_min(1e-6)
        self.lat = [(x - self.mean) / self.std for x in lat]

    def _l9(self, idx: int) -> torch.Tensor:
        item = self.base[idx]
        # expects item["lattice"] as (3,3) (Angstroms)
        L = torch.as_tensor(item["lattice"], dtype=torch.float32).view(-1)[:9]
        return L

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.lat[i]

def get_train_dataset() -> LatticeVecs:
    ds = ds_reg.get_dataset("materials_project_devset_single_task", split="train")
    return LatticeVecs(ds)

# --------------------------
# Tiny epsilon model (MLP) with time embedding
# --------------------------
class EpsMLP(nn.Module):
    def __init__(self, dim: int = 9, hidden: int = 256, t_embed: int = 64):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_embed), nn.SiLU(),
            nn.Linear(t_embed, t_embed), nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(dim + t_embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t is integer timestep; scale to [0,1] and embed
        t = t.float().view(-1, 1) / 1000.0
        h = self.t_embed(t)
        return self.net(torch.cat([x, h], dim=-1))

# --------------------------
# LightningModule using MONAI's DDPMScheduler
# --------------------------
class LatticeDDPM(pl.LightningModule):
    def __init__(
        self,
        dim: int = 9,
        lr: float = 2e-4,
        num_train_timesteps: int = 250,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = EpsMLP(dim=dim)
        # MONAI scheduler handles alphas, noise addition, and reverse steps
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=False,  # keep raw range for vectors
            prediction_type="epsilon"  # network predicts noise
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: torch.Tensor, _):
        x0 = batch  # shape (B, 9)
        B = x0.size(0)
        device = x0.device

        # sample timesteps uniformly
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long
        )

        noise = torch.randn_like(x0)
        # add noise x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise (handled internally)
        xt = self.scheduler.add_noise(original_samples=x0, noise=noise, timesteps=timesteps)

        # predict epsilon
        eps_pred = self.net(xt, timesteps)
        loss = torch.mean((noise - eps_pred) ** 2)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, n: int = 16) -> torch.Tensor:
        """Generate n samples via DDPM reverse process (epsilon prediction)."""
        self.net.eval()
        device = next(self.net.parameters()).device
        x = torch.randn(n, 9, device=device)

        # use the scheduler to step from T-1 -> 0
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self.net(x, t_batch)
            step_out = self.scheduler.step(model_output=eps, timestep=t_batch, sample=x)
            x = step_out.prev_sample  # scheduler returns the previous x_{t-1}
        return x

# --------------------------
# Train + sample
# --------------------------
def main():
    pl.seed_everything(7)
    base = get_train_dataset()
    num_workers = min(8, os.cpu_count() or 4)
    dl = DataLoader(base, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = LatticeDDPM(
        dim=9,
        lr=2e-4,
        num_train_timesteps=250,  # short for speed
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )
    trainer.fit(model, dl)

    # Generate and unnormalize to 3x3 lattices
    samples_z = model.sample(n=16).cpu()
    samples = samples_z * base.std + base.mean
    L = samples.view(-1, 3, 3)  # (N,3,3)
    torch.save(L, "demo_ddpm_lattices.pt")
    print("Saved 3x3 lattice samples to demo_ddpm_lattices.pt")

if __name__ == "__main__":
    main()
