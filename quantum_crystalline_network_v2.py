
# quantum_crystalline_network_v2.py
# ==============================================================================
# QUANTUM CRYSTALLINE NEURAL NETWORK (QCNN) — V2 (lean, temporal, cache-friendly)
# ==============================================================================
#
# What's new vs v1:
# - Real temporal sequences (T>1) so "TemporalCrystal" learns meaningful patterns
# - Block-wise crystallization: entire channel blocks freeze (no grads) when stable
# - Lightweight architecture (no fractal routers); ~10-20x fewer params than v1
# - Optional Fourier features to expose frequencies (inspired by LFFT)
# - Clear speed/accuracy benchmark vs a lean MLP
#
# NOTE: This is still a *concept* model—designed to be readable and fast enough to test.
#
# ==============================================================================

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def banner(txt: str):
    print("\n" + "=" * 80)
    print(txt)
    print("=" * 80 + "\n")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------
# Synthetic temporal dataset
# ------------------------------------------------------------------------------

@dataclass
class DataCfg:
    n_train: int = 4000
    n_val: int = 1000
    seq_len: int = 32
    input_dim: int = 8
    noise_std: float = 0.05
    seed: int = 1234


def make_temporal_regression(cfg: DataCfg):
    """
    Build a non-trivial time series:
    y_t = sum_i a_i * sin(f_i * x_ti + phi_i) + interactions across time lags
    Target is one-step-ahead value for channel 0 (regression).
    """
    g = torch.Generator().manual_seed(cfg.seed)

    T_total = cfg.n_train + cfg.n_val + cfg.seq_len + 1
    X = torch.randn(T_total, cfg.input_dim, generator=g)

    # Hidden periodic drivers per feature
    a = torch.randn(cfg.input_dim, generator=g) * 0.6
    f = torch.rand(cfg.input_dim, generator=g) * 2.5 + 0.5
    phi = torch.rand(cfg.input_dim, generator=g) * 2 * math.pi

    # Construct latent signal
    t = torch.arange(T_total, dtype=torch.float32)
    signal = torch.zeros(T_total)
    for i in range(cfg.input_dim):
        signal += a[i] * torch.sin(f[i] * (X[:, i] + 0.2 * torch.sin(0.1 * t)) + phi[i])

    # Add a lagged interaction term so temporal structure matters
    lag = 3
    interacted = torch.zeros_like(signal)
    interacted[lag:] = signal[:-lag] * 0.3 + 0.2 * torch.sin(signal[:-lag])

    # Older Torch builds don't support generator= in randn_like; omit it for compatibility.
    y = interacted + cfg.noise_std * torch.randn_like(interacted)

    # Build (B, L, D) sequences predicting next-step y for time index
    def make_split(N):
        xs, ys = [], []
        for start in range(N):
            s = start
            e = start + cfg.seq_len
            xs.append(X[s:e])
            ys.append(y[e])  # predict next step after the window
        return torch.stack(xs), torch.stack(ys).unsqueeze(-1)

    X_all, y_all = make_split(cfg.n_train + cfg.n_val)
    X_train, y_train = X_all[:cfg.n_train], y_all[:cfg.n_train]
    X_val, y_val = X_all[cfg.n_train:], y_all[cfg.n_train:]

    return X_train, y_train, X_val, y_val


# ------------------------------------------------------------------------------
# Fourier features (optional) — simple, fast
# ------------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, out_dim) * 2.0, requires_grad=False)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        proj = 2 * math.pi * torch.matmul(x, self.B)  # [B, L, out_dim]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B, L, 2*out_dim]


# ------------------------------------------------------------------------------
# Block-wise crystalline linear layer
# ------------------------------------------------------------------------------

class CrystallineLinear(nn.Module):
    """
    Linear with block-wise crystallization:
    - Channels are partitioned into blocks (e.g., 16-wide)
    - Each block tracks grad magnitude EMA; if stable+used => freeze
    - Frozen blocks stop accumulating grads (detach) and use a cached weight
    """
    def __init__(self, in_dim, out_dim, block: int = 16, bias=True):
        super().__init__()
        self.in_dim, self.out_dim, self.block = in_dim, out_dim, block
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        n_blocks = (out_dim + block - 1) // block
        self.register_buffer("frozen", torch.zeros(n_blocks, dtype=torch.bool))
        self.register_buffer("grad_ema", torch.zeros(n_blocks))
        self.register_buffer("use_ema", torch.zeros(n_blocks))
        self.register_buffer("cache_weight", torch.zeros_like(self.weight))
        if bias:
            self.register_buffer("cache_bias", torch.zeros_like(self.bias))

        self.decay = 0.95
        self.freeze_threshold = 1e-4  # avg grad threshold
        self.use_threshold = 0.1      # activation usage threshold

    def forward(self, x: torch.Tensor):
        # Track usage by block via mean activation magnitude at output
        # We'll approximate usage a posteriori (after matmul) to keep it cheap.
        # Compute with current weight, but detach frozen blocks
        W = self.weight
        b = self.bias

        # Build effective weight mixing cached for frozen blocks
        if self.frozen.any():
            W_eff = W.clone()
            for bi, fr in enumerate(self.frozen):
                if fr:
                    s = bi * self.block
                    e = min((bi + 1) * self.block, self.out_dim)
                    W_eff[s:e] = self.cache_weight[s:e].detach()
                    if b is not None:
                        b = b.clone()
                        b[s:e] = self.cache_bias[s:e].detach()
        else:
            W_eff = W

        y = F.linear(x, W_eff, b)

        # Update usage EMA per block (no grad)
        with torch.no_grad():
            B, L, D = y.shape if y.dim() == 3 else (y.shape[0], 1, y.shape[-1])
            y_abs = y.abs().mean(dim=(0, 1))  # [out_dim]
            # Reduce to blocks
            for bi in range(self.frozen.numel()):
                s = bi * self.block
                e = min((bi + 1) * self.block, self.out_dim)
                m = y_abs[s:e].mean()
                self.use_ema[bi] = self.decay * self.use_ema[bi] + (1 - self.decay) * m

        return y

    def post_backward_update(self):
        # Called after backward() to update grad EMA and freeze blocks when stable
        with torch.no_grad():
            if self.weight.grad is None:
                return
            g = self.weight.grad  # [out, in]
            g_ch = g.pow(2).mean(dim=1).sqrt()  # [out]
            for bi in range(self.frozen.numel()):
                if self.frozen[bi]:
                    continue
                s = bi * self.block
                e = min((bi + 1) * self.block, self.out_dim)
                g_mean = g_ch[s:e].mean()
                self.grad_ema[bi] = self.decay * self.grad_ema[bi] + (1 - self.decay) * g_mean

                # Freeze policy: low grad + decent usage
                if self.grad_ema[bi] < self.freeze_threshold and self.use_ema[bi] > self.use_threshold:
                    self.frozen[bi] = True
                    self.cache_weight[s:e] = self.weight.data[s:e]
                    if self.bias is not None:
                        self.cache_bias[s:e] = self.bias.data[s:e]

    def num_frozen(self):
        return int(self.frozen.sum().item())

    def frac_frozen(self):
        return self.frozen.float().mean().item()


# ------------------------------------------------------------------------------
# Temporal Crystal (lightweight)
# ------------------------------------------------------------------------------

class TemporalCrystal(nn.Module):
    def __init__(self, d_model: int, n_crystals: int = 4):
        super().__init__()
        self.freqs = nn.Parameter(torch.linspace(0.5, 3.5, n_crystals), requires_grad=False)
        self.phase = nn.Parameter(torch.rand(n_crystals) * 2 * math.pi, requires_grad=True)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, t_scalar: int):
        # x: [B, L, D]
        B, L, D = x.shape
        t = torch.tensor(float(t_scalar), device=x.device)
        # Periodic activations
        act = torch.cos(self.freqs * (t / max(L, 1)) + self.phase)  # [C]
        s = torch.sigmoid(act).mean()  # scalar
        return x + s * torch.tanh(self.gate(x))


# ------------------------------------------------------------------------------
# QCNN V2
# ------------------------------------------------------------------------------

class QCNNv2(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, ff_dim: int = 64,
                 n_layers: int = 3, use_fourier: bool = True, fourier_dim: int = 16):
        super().__init__()
        self.use_fourier = use_fourier

        self.in_proj = nn.Linear(input_dim, d_model)
        self.ff = FourierFeatures(d_model, out_dim=fourier_dim) if use_fourier else None

        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_ch = (2 * fourier_dim) if (use_fourier and i == 0) else d_model
            self.layers.append(nn.ModuleDict({
                "cryst1": CrystallineLinear(in_ch, d_model),
                "twist": TemporalCrystal(d_model, n_crystals=4),
                "cryst2": CrystallineLinear(d_model, d_model),
                "norm": nn.LayerNorm(d_model),
            }))

        self.head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 1)
        )

    def forward(self, x: torch.Tensor, timestep: int):
        # x: [B, L, D_in]
        h = self.in_proj(x)  # [B, L, d_model]
        if self.use_fourier:
            h = self.ff(h)  # first block expects 2*fourier_dim

        for li, blk in enumerate(self.layers):
            h1 = blk["cryst1"](h)
            h1 = F.gelu(h1)
            h1 = blk["twist"](h1, timestep)
            h2 = blk["cryst2"](h1)
            h = blk["norm"](h1 + F.gelu(h2))  # residual within the block; dimensions match

        # Pool over time then regress next-step target
        h_pool = h.mean(dim=1)  # [B, D]
        return self.head(h_pool)

    def post_backward_update(self):
        for blk in self.layers:
            blk["cryst1"].post_backward_update()
            blk["cryst2"].post_backward_update()

    def frac_frozen(self):
        fr = []
        for blk in self.layers:
            fr.append(blk["cryst1"].frac_frozen())
            fr.append(blk["cryst2"].frac_frozen())
        return float(sum(fr) / max(1, len(fr)))


# ------------------------------------------------------------------------------
# Baseline MLP (temporal)
# ------------------------------------------------------------------------------

class TemporalMLP(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, ff_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, timestep: Optional[int] = None):
        # Flatten time into features
        B, L, D = x.shape
        xf = x.reshape(B, L * D)
        return self.net(xf)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

@dataclass
class TrainCfg:
    batch: int = 128
    epochs: int = 60
    lr: float = 3e-3
    wd: float = 1e-4


def make_loader(X: torch.Tensor, y: torch.Tensor, batch: int):
    N = X.shape[0]
    def it():
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            yield X[idx].to(DEVICE), y[idx].to(DEVICE)
    return it


def train_one(model: nn.Module, Xtr, ytr, Xval, yval, cfg: TrainCfg, is_qcnn: bool):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    best_val = float("inf")
    t_loader = make_loader(Xtr, ytr, cfg.batch)
    v_loader = make_loader(Xval, yval, cfg.batch)

    for ep in range(cfg.epochs):
        model.train()
        total = 0.0
        nb = 0
        for xb, yb in t_loader():
            opt.zero_grad(set_to_none=True)
            if is_qcnn:
                pred = model(xb, timestep=ep)
            else:
                pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            # Freeze updates for qcnn
            if is_qcnn:
                model.post_backward_update()
            opt.step()
            total += loss.item()
            nb += 1

        # val
        model.eval()
        vs = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in v_loader():
                if is_qcnn:
                    pv = model(xb, timestep=ep)
                else:
                    pv = model(xb)
                vs += F.mse_loss(pv, yb).item()
                vn += 1
        val = vs / max(1, vn)
        best_val = min(best_val, val)

        if (ep + 1) % 10 == 0:
            cry = getattr(model, "frac_frozen", lambda: 0.0)()
            print(f"  Epoch {ep+1:02d}/{cfg.epochs} | Train {total/max(1,nb):.4f} | Val {val:.4f} | Frozen {cry*100:5.2f}%")

    return best_val


@torch.no_grad()
def bench_ms(model: nn.Module, sample: torch.Tensor, calls: int = 100):
    """
    Robust micro-benchmark that supports models with or without a `timestep` kwarg.
    """
    model.eval().to(DEVICE)

    def call_once():
        try:
            return model(sample, timestep=0)
        except TypeError:
            return model(sample)

    # warmup
    for _ in range(20):
        _ = call_once()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(calls):
            _ = call_once()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / calls
    else:
        t0 = time.perf_counter()
        for _ in range(calls):
            _ = call_once()
        return (time.perf_counter() - t0) * 1000 / calls


# ------------------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------------------

def main():
    banner("QCNN v2 — Temporal Regression Demo")

    # Data
    cfg = DataCfg()
    Xtr, ytr, Xv, yv = make_temporal_regression(cfg)

    # Models
    mlp = TemporalMLP(input_dim=cfg.input_dim * cfg.seq_len, d_model=64, ff_dim=64)
    qcnn = QCNNv2(input_dim=cfg.input_dim, d_model=48, ff_dim=64, n_layers=3, use_fourier=True, fourier_dim=12)

    print("Parameters:")
    print(f"  TemporalMLP : {count_params(mlp):,}")
    print(f"  QCNNv2      : {count_params(qcnn):,}")

    # Train
    banner("TRAINING")
    mb = train_one(mlp, Xtr, ytr, Xv, yv, TrainCfg(epochs=50), is_qcnn=False)
    qb = train_one(qcnn, Xtr, ytr, Xv, yv, TrainCfg(epochs=50), is_qcnn=True)

    # Benchmark
    banner("SPEED BENCHMARK (ms/batch)")
    B = 64
    sample_mlp = Xtr[:B].to(DEVICE).reshape(B, cfg.seq_len * cfg.input_dim)
    sample_q = Xtr[:B].to(DEVICE)
    mlp_ms = bench_ms(mlp, sample_mlp)
    q_ms = bench_ms(qcnn, sample_q)
    print(f"  TemporalMLP : {mlp_ms:.3f} ms")
    print(f"  QCNNv2      : {q_ms:.3f} ms")

    # Results
    banner("RESULTS")
    print(f"  MLP  Best Val MSE: {mb:.6f}")
    print(f"  QCNN Best Val MSE: {qb:.6f}")

    if hasattr(qcnn, "frac_frozen"):
        print(f"  QCNN Frozen Blocks: {qcnn.frac_frozen()*100:.2f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()
