# rff_sweep.py
# Simplified RFF vs Baseline with a clear "EXPERIMENTS" section to try different values.
# Windows-safe: avoids Triton/Inductor; uses eager or eager-backend compile when available.
# Adds GPU-correct timing, warm-up, and first-epoch discard for stable speed numbers.

'''
Results:

C:\Users\MAC-USER\PycharmProjects\BetterLearning\.venv\Scripts\python.exe C:\Users\MAC-USER\PycharmProjects\BetterLearning\v1.py
[Info] Device selected: cuda | torch 2.5.1+cu121 | compile=True | amp=True
[Info] Effective backend: eager (compile skipped on Windows)

=== Results: H8_parity ===
exp	seed	model	H	M	epochs	batch	ms_per_epoch	test_mse
H8_parity	101	Baseline	8	-	120	256	0.43647112766233814	0.08856122940778732
H8_parity	101	RFF19+	8	19	120	256	0.8191636096195678	0.03392601013183594
H8_parity	133	Baseline	8	-	120	256	0.41077763493321523	0.12669818103313446
H8_parity	133	RFF19+	8	19	120	256	0.8048861420789021	0.12554189562797546

=== Results: H8_M21 ===
exp	seed	model	H	M	epochs	batch	ms_per_epoch	test_mse
H8_M21	101	Baseline	8	-	120	256	0.4061689196514483	0.0869797021150589
H8_M21	101	RFF19+	8	21	120	256	0.8416869493425723	0.09345987439155579
H8_M21	133	Baseline	8	-	120	256	0.41042276557420154	0.12059777975082397
H8_M21	133	RFF19+	8	21	120	256	0.8078513806607542	0.1273825764656067

=== Results: H16_parity ===
exp	seed	model	H	M	epochs	batch	ms_per_epoch	test_mse
H16_parity	202	Baseline	16	-	120	256	0.41363459842211714	0.10463812947273254
H16_parity	202	RFF19+	16	43	120	256	0.832253130210214	0.10531734675168991
H16_parity	303	Baseline	16	-	120	256	0.40720761857446847	0.10962364077568054
H16_parity	303	RFF19+	16	43	120	256	0.8133500049762028	0.09637745469808578

=== Results: H8_quick ===
exp	seed	model	H	M	epochs	batch	ms_per_epoch	test_mse
H8_quick	999	Baseline	8	-	60	256	0.4326418092695333	0.11355612426996231
H8_quick	999	RFF19+	8	19	60	256	0.953476156892076	0.11612322926521301

=== Results: H8_rff_only ===
exp	seed	model	H	M	epochs	batch	ms_per_epoch	test_mse
H8_rff_only	42	RFF19+	8	19	120	512	0.8538479344183657	0.09207262098789215
H8_rff_only	43	RFF19+	8	19	120	512	0.8020363014285304	0.08145032823085785

Process finished with exit code 0


'''
import math
import time
import csv
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# Device & global config
# ===============================
# Force mode: "auto" | "cuda" | "cpu"
FORCE_DEVICE = "auto"   # change to "cuda" or "cpu" if you want to force it
USE_COMPILE  = True     # we will auto-disable on Windows; elsewhere use backend="eager"
USE_AMP      = True     # AMP (bf16 on CUDA if available)

def get_device():
    if FORCE_DEVICE == "cuda":
        return "cuda"
    if FORCE_DEVICE == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()
DTYPE  = torch.float32

print(f"[Info] Device selected: {DEVICE} | torch {torch.__version__} | compile={USE_COMPILE} | amp={USE_AMP}")

def report_compile_backend():
    backend = "eager (no compile)"
    if hasattr(torch, "compile"):
        if sys.platform.startswith("win"):
            backend = "eager (compile skipped on Windows)"
        else:
            backend = "compile(backend='eager')" if USE_COMPILE else "eager (compile disabled)"
    print(f"[Info] Effective backend: {backend}")

report_compile_backend()

def maybe_compile(model):
    """
    Windows-safe compile wrapper:
      - On Windows, skip compile entirely (Triton/Inductor not reliable).
      - On other OSes, try torch.compile with backend='eager' (no Triton kernels).
      - If anything fails, return the original model.
    """
    if sys.platform.startswith("win"):
        return model
    if not (USE_COMPILE and hasattr(torch, "compile")):
        return model
    try:
        return torch.compile(model, backend="eager")
    except Exception as e:
        print("[WARN] torch.compile(backend='eager') failed; using eager:", e)
        return model

# ===============================
# Data
# ===============================
def make_data(n=900, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    X = torch.rand(n, 1, generator=g).to(device=device, dtype=dtype)
    f = lambda x: torch.sin(6*x) + 0.3*torch.sin(12*x)
    # cross-version safe noise (no generator= for randn_like on CUDA)
    y = f(X) + 0.05 * torch.randn_like(X)
    ntr = int(0.7*n); nva = int(0.15*n)
    idx = torch.randperm(n, generator=g)
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]

# ===============================
# Baseline (1–H–1)
# ===============================
class BaselineMLP(nn.Module):
    """Param count = 3H + 1."""
    def __init__(self, H: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(H, 1) * 0.4)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.randn(1, H) * 0.4)
        self.b2 = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        h = torch.tanh(x @ self.W1.t() + self.b1)
        y = h @ self.W2.t() + self.b2
        return y

# ===============================
# RFF-19+ variant (general H)
# ===============================
def log_uniform_radii(num, rmin=0.1, rmax=20.0, device="cpu", dtype=torch.float32):
    u = torch.rand(num, 1, device=device, dtype=dtype)
    return torch.exp(torch.log(torch.tensor(rmin, dtype=dtype, device=device)) +
                     u * (torch.log(torch.tensor(rmax, dtype=dtype, device=device)) -
                          torch.log(torch.tensor(rmin, dtype=dtype, device=device))))

def random_unit_directions(num, dim=3, device="cpu", dtype=torch.float32):
    x = torch.randn(num, dim, device=device, dtype=dtype)
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)

@dataclass
class RFFConfig:
    H: int
    M: int
    rmin: float = 0.1
    rmax: float = 20.0
    dtype: torch.dtype = torch.float32

class RFF19Plus(nn.Module):
    """
    Trainables: c (M,), Mmix (2x2), rW1, rW2
    Fixed: Wfreq (M,3), bfreq (M,), cached Phi blocks on device
    """
    def __init__(self, cfg: RFFConfig, device="cpu"):
        super().__init__()
        H, M = cfg.H, cfg.M
        self.H, self.M = H, M
        self.device = device
        self.dtype = cfg.dtype

        dirs  = random_unit_directions(M, dim=3, device=device, dtype=cfg.dtype)
        radii = log_uniform_radii(M, cfg.rmin, cfg.rmax, device=device, dtype=cfg.dtype)
        self.register_buffer("Wfreq", dirs * radii)  # (M,3)
        self.register_buffer("bfreq", torch.rand(M, device=device, dtype=cfg.dtype) * 2*math.pi)

        self.c    = nn.Parameter(torch.randn(M, device=device, dtype=cfg.dtype) * 0.2)
        self.Mmix = nn.Parameter(torch.randn(2, 2, device=device, dtype=cfg.dtype) * 0.1)
        self.rW1  = nn.Parameter(torch.tensor(0.0, device=device, dtype=cfg.dtype))
        self.rW2  = nn.Parameter(torch.tensor(0.0, device=device, dtype=cfg.dtype))

        # Cache Phi blocks
        I_W1, J_W1 = torch.meshgrid(torch.arange(H, device=device, dtype=cfg.dtype),
                                    torch.arange(1, device=device, dtype=cfg.dtype), indexing="ij")
        I_b1 = torch.arange(H, device=device, dtype=cfg.dtype); J_b1 = torch.zeros(H, device=device, dtype=cfg.dtype)
        I_W2, J_W2 = torch.meshgrid(torch.arange(1, device=device, dtype=cfg.dtype),
                                    torch.arange(H, device=device, dtype=cfg.dtype), indexing="ij")
        I_b2 = torch.arange(1, device=device, dtype=cfg.dtype); J_b2 = torch.zeros(1, device=device, dtype=cfg.dtype)

        self.register_buffer("Phi_W1", self._phi(I_W1, J_W1, layer_id=1))
        self.register_buffer("Phi_b1", self._phi(I_b1, J_b1, layer_id=1)[:H, :])
        self.register_buffer("Phi_W2", self._phi(I_W2, J_W2, layer_id=2))
        self.register_buffer("Phi_b2", self._phi(I_b2, J_b2, layer_id=2)[:1, :])

    def _phi(self, I, J, layer_id:int):
        IJL = torch.stack([I, J, torch.full_like(I, float(layer_id))], dim=-1).reshape(-1, 3)
        return torch.cos(IJL @ self.Wfreq.t() + self.bfreq)

    def _scales(self, layer_id:int):
        e = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype) if layer_id==1 \
            else torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype)
        return (self.Mmix @ e)  # (2,) = [sw, sb]

    def forward(self, x):
        gW1 = (self.Phi_W1 @ self.c).reshape(self.H, 1)
        gb1 = (self.Phi_b1 @ self.c).reshape(self.H)
        gW2 = (self.Phi_W2 @ self.c).reshape(1, self.H)
        gb2 = (self.Phi_b2 @ self.c).reshape(1)

        sw1, sb1 = self._scales(1)
        sw2, sb2 = self._scales(2)

        W1 = sw1 * gW1 + self.rW1
        b1 = sb1 * gb1
        W2 = sw2 * gW2 + self.rW2
        b2 = sb2 * gb2

        h = torch.tanh(x @ W1.t() + b1)
        y = h @ W2.t() + b2
        return y

# ===============================
# Training / Eval
# ===============================
def cosine_lr(step, total_steps, base_lr, warmup=10, min_lr=1e-4):
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    prog = (step - warmup) / max(1, (total_steps - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_lr + (base_lr - min_lr) * cos

def train_epoch(model, opt, X, y, batch, base_lr, steps_done, total_steps, use_amp=True):
    """
    Accurate GPU timing using CUDA events; CPU falls back to perf_counter.
    Returns (steps_done, ms_per_batch).
    """
    model.train()
    n = X.size(0)
    perm = torch.randperm(n, device=X.device)

    if X.is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    for i in range(0, n, batch):
        j = perm[i:i+batch]
        lr_now = cosine_lr(steps_done, total_steps, base_lr, warmup=10, min_lr=1e-4)
        for g in opt.param_groups:
            g['lr'] = lr_now
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(use_amp and X.is_cuda))
        with ctx:
            yhat = model(X[j])
            loss = F.mse_loss(yhat, y[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps_done += 1

    if X.is_cuda:
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / max(1, (n + batch - 1)//batch)  # ms per batch
    else:
        ms = (time.perf_counter() - t0) * 1000.0 / max(1, (n + batch - 1)//batch)
    return steps_done, ms

@torch.no_grad()
def eval_mse(model, X, y):
    model.eval()
    return F.mse_loss(model(X), y).item()

def param_parity_M(H:int) -> int:
    # Baseline params = 3H + 1; RFF params = M + 4 + 2 => M = 3H - 5
    return max(1, 3*H - 5)

@torch.no_grad()
def warmup_model(model, X, steps=2, batch=64):
    """Tiny forward-only warm-up to trigger CUDA context & kernels before timing."""
    if not X.is_cuda:
        return
    model.train()
    n = min(4 * batch, X.size(0))
    perm = torch.randperm(X.size(0), device=X.device)[:n]
    for _ in range(steps):
        _ = model(X[perm[:batch]])
    torch.cuda.synchronize()

# ===============================
# Experiment runner
# ===============================
def run_experiment(
    name:str,
    H:int,
    M: Optional[int],
    epochs:int=120,
    batch:int=256,
    seeds: List[int] = [1337],
    lr_rff: float = 1.5e-2,
    lr_base: float = 2.0e-2,
    run_baseline: bool = True,
    run_rff: bool = True,
    n_data:int = 900,
    save_csv: Optional[str] = None,
):
    assert run_baseline or run_rff, "Nothing to run: enable baseline and/or rff."
    results = []
    M_eff = param_parity_M(H) if (M is None) else M

    for seed in seeds:
        # Data
        Xtr, ytr, Xva, yva, Xte, yte = make_data(n=n_data, seed=seed, device=DEVICE, dtype=DTYPE)
        steps_total = epochs * math.ceil(Xtr.size(0) / batch)

        # Baseline
        if run_baseline:
            base = BaselineMLP(H).to(DEVICE)
            base = maybe_compile(base)
            warmup_model(base, Xtr)  # GPU warm-up before timing
            optb = torch.optim.AdamW(base.parameters(), lr=lr_base, weight_decay=1e-4)
            steps = 0; ms_list = []
            for ep in range(epochs):
                steps, ms = train_epoch(base, optb, Xtr, ytr, batch, lr_base, steps, steps_total, use_amp=USE_AMP)
                ms_list.append(ms)
            avg_ms = sum(ms_list[1:]) / max(1, len(ms_list)-1)  # drop first epoch (warm-up)
            mse_b = eval_mse(base, Xte, yte)
            results.append({"exp": name, "seed": seed, "model": "Baseline", "H": H, "M": "-", "epochs": epochs,
                            "batch": batch, "ms_per_epoch": avg_ms, "test_mse": mse_b})

        # RFF
        if run_rff:
            rff = RFF19Plus(RFFConfig(H=H, M=M_eff, dtype=DTYPE), device=DEVICE).to(DEVICE)
            rff = maybe_compile(rff)
            warmup_model(rff, Xtr)   # GPU warm-up before timing
            optr = torch.optim.AdamW(rff.parameters(), lr=lr_rff, weight_decay=1e-4)
            steps = 0; ms_list = []
            for ep in range(epochs):
                steps, ms = train_epoch(rff, optr, Xtr, ytr, batch, lr_rff, steps, steps_total, use_amp=USE_AMP)
                ms_list.append(ms)
            avg_ms = sum(ms_list[1:]) / max(1, len(ms_list)-1)  # drop first epoch (warm-up)
            mse_r = eval_mse(rff, Xte, yte)
            results.append({"exp": name, "seed": seed, "model": "RFF19+", "H": H, "M": M_eff, "epochs": epochs,
                            "batch": batch, "ms_per_epoch": avg_ms, "test_mse": mse_r})

    # Aggregate print
    print("\n=== Results:", name, "===")
    hdr = ["exp","seed","model","H","M","epochs","batch","ms_per_epoch","test_mse"]
    print("\t".join(hdr))
    for r in results:
        print("\t".join(str(r[k]) for k in hdr))

    # Optional CSV
    if save_csv:
        with open(save_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if f.tell() == 0:
                w.writeheader()
            for r in results:
                w.writerow(r)

    return results

# ===============================
# EXPERIMENTS — EDIT BELOW
# ===============================
EXPERIMENTS: List[Dict] = [
    # 1) Strict param parity for 1–8–1 (M auto = 3*H-5 = 19)
    dict(name="H8_parity", H=8,  M=None, epochs=120, batch=256, seeds=[101,133], run_baseline=True,  run_rff=True),

    # 2) Same H=8 but CUSTOM M (e.g., 21) to see effect on accuracy/speed
    dict(name="H8_M21",    H=8,  M=21,   epochs=120, batch=256, seeds=[101,133], run_baseline=True,  run_rff=True),

    # 3) Wider: 1–16–1 with parity (M=3*16-5=43)
    dict(name="H16_parity",H=16, M=None, epochs=120, batch=256, seeds=[202,303], run_baseline=True,  run_rff=True),

    # 4) Quick smoke (short epochs)
    dict(name="H8_quick",  H=8,  M=None, epochs=60,  batch=256, seeds=[999],    run_baseline=True,  run_rff=True),

    # 5) Only RFF (no baseline), custom bigger batch
    dict(name="H8_rff_only",
         H=8, M=None, epochs=120, batch=512, seeds=[42,43], run_baseline=False, run_rff=True),
]

def main():
    for cfg in EXPERIMENTS:
        run_experiment(**cfg)

if __name__ == "__main__":
    # Optional: clear CUDA cache between runs (helps timing consistency on GPU)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    main()
