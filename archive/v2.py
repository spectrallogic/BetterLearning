# rff_sweep_v2_5.py
# V2.5: Accuracy-recovering FWRFF:
# - rank_residual up to 4 (more expressive at tiny cost)
# - M & Octave sweeps (19/27/35; O=2/3)
# - gentler gate L1 (1e-4)
# - toggle fp16 vs fp32 storage for Φ (phi_store_fp16)
# - Windows-safe compile, CUDA-event timing, early stop, warmup
'''
Results of current code:
C:\Users\MAC-USER\PycharmProjects\BetterLearning\.venv\Scripts\python.exe C:\Users\MAC-USER\PycharmProjects\BetterLearning\v2.py
[Info] Device selected: cuda | torch 2.5.1+cu121 | compile=True | amp=True
[Info] Effective backend: eager (compile skipped on Windows)

=== Results: H8_baseline ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_baseline	101	Baseline	8	-	116	256	0.8207244973251787	0.09172206372022629
H8_baseline	133	Baseline	8	-	43	256	0.6993902212097532	0.118233323097229
H8_baseline	202	Baseline	8	-	120	256	0.4238624980135792	0.10151109844446182

=== Results: H8_sine ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_sine	101	SineMLP	8	-	120	256	0.47399197740047244	0.018153423443436623
H8_sine	133	SineMLP	8	-	82	256	0.4593937127187908	0.0031592482700943947
H8_sine	202	SineMLP	8	-	120	256	0.45426429286390474	0.019657887518405914

=== Results: H8_wave ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_wave	101	WaveBasisMLP	8	-	120	256	0.5104224982381869	0.003542523365467787
H8_wave	133	WaveBasisMLP	8	-	120	256	0.5232718869083735	0.008469806052744389
H8_wave	202	WaveBasisMLP	8	-	45	256	0.5639883622978674	0.0025728298351168633

=== Results: H8_fwrff_parity_O2_rank4 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_fwrff_parity_O2_rank4	101	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	19	78	512	1.2726181791974354	0.03243417292833328
H8_fwrff_parity_O2_rank4	133	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	19	106	512	1.2178480738685244	0.036721061915159225
H8_fwrff_parity_O2_rank4	202	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	19	96	512	1.2308407557638068	0.037930168211460114

=== Results: H8_fwrff_parity_O3_rank4 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_fwrff_parity_O3_rank4	101	FractalWaveRFF(O=3,sine,warp=True,rank=4,phi16=True)	8	19	91	512	1.2837831126319037	0.03163224086165428
H8_fwrff_parity_O3_rank4	133	FractalWaveRFF(O=3,sine,warp=True,rank=4,phi16=True)	8	19	95	512	1.230118639925693	0.031118569895625114
H8_fwrff_parity_O3_rank4	202	FractalWaveRFF(O=3,sine,warp=True,rank=4,phi16=True)	8	19	95	512	1.22926076295528	0.03798087686300278

=== Results: H8_fwrff_M27_O2 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_fwrff_M27_O2	101	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	27	100	512	1.2863418201003411	0.0284289438277483
H8_fwrff_M27_O2	133	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	27	89	512	1.231372359124097	0.03376910090446472
H8_fwrff_M27_O2	202	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	27	93	512	1.3516140839327937	0.035888995975255966

=== Results: H8_fwrff_M35_O2 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_fwrff_M35_O2	101	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	35	120	512	1.3675946247677844	0.03111639991402626
H8_fwrff_M35_O2	133	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	35	91	512	1.2421391963958741	0.03340974822640419
H8_fwrff_M35_O2	202	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	8	35	80	512	1.222458528566964	0.03536166623234749

=== Results: H8_fwrff_parity_O2_phi32 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H8_fwrff_parity_O2_phi32	101	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=False)	8	19	99	512	1.1724318399721263	0.03093881532549858
H8_fwrff_parity_O2_phi32	133	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=False)	8	19	91	512	1.1268702189127604	0.03743079677224159
H8_fwrff_parity_O2_phi32	202	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=False)	8	19	93	512	1.1580879986286163	0.036121729761362076

=== Results: H16_baseline ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H16_baseline	202	Baseline	16	-	27	256	0.43099076625628346	0.10710566490888596
H16_baseline	303	Baseline	16	-	110	256	0.4490582505497364	0.10774406790733337

=== Results: H16_sine ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H16_sine	202	SineMLP	16	-	70	256	0.4892721176147463	0.002116117626428604
H16_sine	303	SineMLP	16	-	101	256	0.5026270922025046	0.002464559394866228

=== Results: H16_wave ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H16_wave	202	WaveBasisMLP	16	-	39	256	0.5067609820449561	0.003198782214894891
H16_wave	303	WaveBasisMLP	16	-	43	256	0.5073358747694228	0.0029388873372226954

=== Results: H16_fwrff_parity_O2_rank4 ===
exp	seed	model	H	M	epochs_used	batch	ms_per_epoch	test_mse
H16_fwrff_parity_O2_rank4	202	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	16	43	80	512	1.3134069895442528	0.037753719836473465
H16_fwrff_parity_O2_rank4	303	FractalWaveRFF(O=2,sine,warp=True,rank=4,phi16=True)	16	43	103	512	1.2355220387963688	0.033286191523075104


'''

import math, time, csv, sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

# ===== Global =====
FORCE_DEVICE = "auto"   # "auto" | "cuda" | "cpu"
USE_COMPILE  = True     # skipped on Windows; else backend="eager"
USE_AMP      = True
EARLY_STOP_PATIENCE = 15

def get_device():
    if FORCE_DEVICE == "cuda": return "cuda"
    if FORCE_DEVICE == "cpu":  return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()
DTYPE  = torch.float32
print(f"[Info] Device selected: {DEVICE} | torch {torch.__version__} | compile={USE_COMPILE} | amp={USE_AMP}")

def report_compile_backend():
    backend = "eager (no compile)"
    if hasattr(torch, "compile"):
        if sys.platform.startswith("win"): backend = "eager (compile skipped on Windows)"
        else: backend = "compile(backend='eager')" if USE_COMPILE else "eager (compile disabled)"
    print(f"[Info] Effective backend: {backend}")
report_compile_backend()

def maybe_compile(model):
    if sys.platform.startswith("win"): return model
    if not (USE_COMPILE and hasattr(torch, "compile")): return model
    try: return torch.compile(model, backend="eager")
    except Exception as e:
        print("[WARN] torch.compile(backend='eager') failed; using eager:", e)
        return model

# ===== Data =====
def make_data(n=900, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    X = torch.rand(n, 1, generator=g).to(device=device, dtype=dtype)
    f = lambda x: torch.sin(6*x) + 0.3*torch.sin(12*x)
    y = f(X) + 0.05 * torch.randn_like(X)
    ntr = int(0.7*n); nva = int(0.15*n)
    idx = torch.randperm(n, generator=g)
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]

# ===== Baselines =====
class BaselineMLP(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(H, 1) * 0.4)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.randn(1, H) * 0.4)
        self.b2 = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        h = torch.tanh(x @ self.W1.t() + self.b1)
        return h @ self.W2.t() + self.b2

class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega0=15.0):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim); self.omega0 = omega0
        with torch.no_grad():
            val = math.sqrt(6/in_dim)/omega0
            self.W.weight.uniform_(-val, val); self.W.bias.uniform_(-val, val)
    def forward(self, x): return torch.sin(self.omega0 * self.W(x))

class SineMLP(nn.Module):
    def __init__(self, H: int, omega0=15.0):
        super().__init__()
        self.sin1 = SineLayer(1, H, omega0=omega0)
        self.out  = nn.Linear(H, 1)
        with torch.no_grad():
            self.out.weight.uniform_(-1e-2, 1e-2); self.out.bias.zero_()
    def forward(self, x): return self.out(self.sin1(x))

class WaveBasisMLP(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.a = nn.Parameter(torch.randn(H, 1) * 5.0)
        self.phi = nn.Parameter(torch.zeros(H))
        self.out = nn.Linear(2*H, 1)
        with torch.no_grad():
            self.out.weight.uniform_(-1e-2, 1e-2); self.out.bias.zero_()
    def forward(self, x):
        z = x @ self.a.t() + self.phi
        h = torch.cat([torch.sin(z), torch.cos(z)], dim=1)
        return self.out(h)

# ===== FWRFF =====
def log_uniform_radii(num, rmin=0.1, rmax=20.0, device="cpu", dtype=torch.float32):
    u = torch.rand(num, 1, device=device, dtype=dtype)
    return torch.exp(torch.log(torch.tensor(rmin, dtype=dtype, device=device)) +
                     u * (torch.log(torch.tensor(rmax, dtype=dtype, device=device)) -
                          torch.log(torch.tensor(rmin, dtype=dtype, device=device))))

def random_unit_dirs(num, dim=3, device="cpu", dtype=torch.float32):
    x = torch.randn(num, dim, device=device, dtype=dtype)
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)

@dataclass
class FWRFFConfig:
    H: int
    M: int
    octaves: int = 2
    octave_ratio: float = 1.7
    rmin: float = 0.05
    rmax: float = 25.0
    gate_l1: float = 1e-4          # gentler
    hidden_act: str = "sine"       # "tanh" | "sine"
    dtype: torch.dtype = torch.float32
    rank_residual: int = 4         # more expressive
    warp_enabled: bool = True
    phi_store_fp16: bool = True    # toggle Φ storage dtype (fp16 on CUDA vs fp32)

class FractalWaveRFF(nn.Module):
    def __init__(self, cfg: FWRFFConfig, device="cpu"):
        super().__init__()
        self.cfg = cfg
        H, M, O = cfg.H, cfg.M, cfg.octaves
        self.H, self.M, self.O = H, M, O
        self.device = device; self.dtype = cfg.dtype

        dirs  = random_unit_dirs(M, dim=3, device=device, dtype=cfg.dtype)
        radii = log_uniform_radii(M, cfg.rmin, cfg.rmax, device=device, dtype=cfg.dtype)
        self.register_buffer("Wfreq_base", dirs * radii)
        self.register_buffer("bfreq_base", torch.rand(M, device=device, dtype=cfg.dtype) * 2*math.pi)

        self.c     = nn.Parameter(torch.randn(O, M, device=device, dtype=cfg.dtype) * 0.1)
        self.gate  = nn.Parameter(torch.full((M,), 0.0, device=device, dtype=cfg.dtype))
        self.Mmix1 = nn.Parameter(torch.randn(O, 2, device=device, dtype=cfg.dtype) * 0.1)
        self.Mmix2 = nn.Parameter(torch.randn(O, 2, device=device, dtype=cfg.dtype) * 0.1)
        self.rW1   = nn.Parameter(torch.zeros(1, device=device, dtype=cfg.dtype))
        self.rW2   = nn.Parameter(torch.zeros(1, device=device, dtype=cfg.dtype))

        r = cfg.rank_residual
        self.W1_u = nn.Parameter(torch.randn(H, r, device=device, dtype=cfg.dtype) * 1e-2)
        self.W1_v = nn.Parameter(torch.randn(r, 1, device=device, dtype=cfg.dtype) * 1e-2)
        self.W2_u = nn.Parameter(torch.randn(1, r, device=device, dtype=cfg.dtype) * 1e-2)
        self.W2_v = nn.Parameter(torch.randn(r, H, device=device, dtype=cfg.dtype) * 1e-2)

        self.warp_a = nn.Parameter(torch.tensor(1.0, device=device, dtype=cfg.dtype))
        self.warp_b = nn.Parameter(torch.tensor(0.0, device=device, dtype=cfg.dtype))
        self.warp_amp = nn.Parameter(torch.tensor(0.05, device=device, dtype=cfg.dtype))
        self.warp_w   = nn.Parameter(torch.tensor(8.0, device=device, dtype=cfg.dtype))
        self.warp_phi = nn.Parameter(torch.tensor(0.0, device=device, dtype=cfg.dtype))

        # parameter coordinates
        I_W1, J_W1 = torch.meshgrid(torch.arange(H, device=device, dtype=cfg.dtype),
                                    torch.arange(1, device=device, dtype=cfg.dtype), indexing="ij")
        I_b1 = torch.arange(H, device=device, dtype=cfg.dtype); J_b1 = torch.zeros(H, device=device, dtype=cfg.dtype)
        I_W2, J_W2 = torch.meshgrid(torch.arange(1, device=device, dtype=cfg.dtype),
                                    torch.arange(H, device=device, dtype=cfg.dtype), indexing="ij")
        I_b2 = torch.arange(1, device=device, dtype=cfg.dtype); J_b2 = torch.zeros(1, device=device, dtype=cfg.dtype)
        IJ_W1 = torch.stack([I_W1, J_W1, torch.full_like(I_W1, 1.0)], dim=-1).reshape(-1,3)
        IJ_b1 = torch.stack([I_b1, J_b1, torch.full_like(I_b1, 1.0)], dim=-1).reshape(-1,3)
        IJ_W2 = torch.stack([I_W2, J_W2, torch.full_like(I_W2, 2.0)], dim=-1).reshape(-1,3)
        IJ_b2 = torch.stack([I_b2, J_b2, torch.full_like(I_b2, 2.0)], dim=-1).reshape(-1,3)
        self.register_buffer("IJ_W1", IJ_W1)
        self.register_buffer("IJ_b1", IJ_b1)
        self.register_buffer("IJ_W2", IJ_W2)
        self.register_buffer("IJ_b2", IJ_b2)

        # precompute Φ (√M norm)
        scales = (cfg.octave_ratio ** torch.arange(cfg.octaves, device=device, dtype=cfg.dtype))
        self.register_buffer("scales", scales)
        W_scaled = self.Wfreq_base.unsqueeze(0) * scales.view(cfg.octaves, 1, 1)
        b = self.bfreq_base.view(1, 1, M)

        Phi_W1 = torch.cos(torch.einsum('pc,omc->opm', self.IJ_W1, W_scaled) + b)   # (O, H, M)
        Phi_b1 = torch.cos(torch.einsum('pc,omc->opm', self.IJ_b1, W_scaled) + b)   # (O, H, M)
        Phi_W2 = torch.cos(torch.einsum('pc,omc->opm', self.IJ_W2, W_scaled) + b)   # (O, H, M)
        Phi_b2 = torch.cos(torch.einsum('pc,omc->opm', self.IJ_b2, W_scaled) + b)   # (O, 1, M)

        scale_M = (float(M) ** 0.5)
        Phi_W1, Phi_b1, Phi_W2, Phi_b2 = [x/scale_M for x in (Phi_W1, Phi_b1, Phi_W2, Phi_b2)]

        # storage dtype for Φ
        if device == "cuda" and cfg.phi_store_fp16:
            phi_dtype = torch.float16
        else:
            phi_dtype = torch.float32
        self.register_buffer("Phi_W1", Phi_W1.to(dtype=phi_dtype, device=device))
        self.register_buffer("Phi_b1", Phi_b1.to(dtype=phi_dtype, device=device))
        self.register_buffer("Phi_W2", Phi_W2.to(dtype=phi_dtype, device=device))
        self.register_buffer("Phi_b2", Phi_b2.to(dtype=phi_dtype, device=device))
        self.phi_dtype = phi_dtype

    def _warp(self, x):
        if not self.cfg.warp_enabled: return x
        return self.warp_a * x + self.warp_b + self.warp_amp * torch.sin(self.warp_w * x + self.warp_phi)

    def _synthesize_params(self):
        O, H, M = self.O, self.H, self.M
        gates = torch.sigmoid(self.gate).to(self.phi_dtype)            # (M,)
        c_eff = (self.c.to(self.phi_dtype) * gates.view(1, M)).unsqueeze(-1)  # (O,M,1)

        W1_oct = torch.bmm(self.Phi_W1, c_eff).view(O, H, 1)
        b1_oct = torch.bmm(self.Phi_b1, c_eff).view(O, H)
        W2_oct = torch.bmm(self.Phi_W2, c_eff).view(O, 1, H)
        b2_oct = torch.bmm(self.Phi_b2, c_eff).view(O, 1)

        s1 = self.Mmix1.to(self.phi_dtype); s2 = self.Mmix2.to(self.phi_dtype)
        gW1 = (W1_oct * s1[:, 0].view(O, 1, 1)).sum(dim=0)
        gb1 = (b1_oct * s1[:, 1].view(O, 1)).sum(dim=0)
        gW2 = (W2_oct * s2[:, 0].view(O, 1, 1)).sum(dim=0)
        gb2 = (b2_oct * s2[:, 1].view(O, 1)).sum(dim=0)

        gW1 = gW1.to(self.dtype); gb1 = gb1.to(self.dtype)
        gW2 = gW2.to(self.dtype); gb2 = gb2.to(self.dtype)

        W1 = gW1 + self.rW1 + (self.W1_u @ self.W1_v)                # (H,1)
        W2 = gW2 + self.rW2 + (self.W2_u @ self.W2_v)                # (1,H)
        b1 = gb1
        b2 = gb2
        return W1, b1, W2, b2

    def forward(self, x):
        xw = self._warp(x)
        W1, b1, W2, b2 = self._synthesize_params()
        if self.cfg.hidden_act == "sine":
            h = torch.sin(xw @ W1.t() + b1)
        else:
            h = torch.tanh(xw @ W1.t() + b1)
        return h @ W2.t() + b2

    def gate_l1_loss(self):
        return self.cfg.gate_l1 * torch.sigmoid(self.gate).abs().mean()

# ===== Train/Eval =====
def cosine_lr(step, total_steps, base_lr, warmup=10, min_lr=1e-4):
    if step < warmup: return base_lr * float(step + 1) / float(warmup)
    prog = (step - warmup) / max(1, (total_steps - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_lr + (base_lr - min_lr) * cos

def train_epoch(model, opt, X, y, batch, base_lr, steps_done, total_steps,
                use_amp=True, extra_loss_fn=None):
    model.train(); n = X.size(0); perm = torch.randperm(n, device=X.device)
    if X.is_cuda:
        torch.cuda.synchronize(); start = torch.cuda.Event(True); end = torch.cuda.Event(True); start.record()
    else:
        t0 = time.perf_counter()
    for i in range(0, n, batch):
        j = perm[i:i+batch]
        lr_now = cosine_lr(steps_done, total_steps, base_lr, 10, 1e-4)
        for g in opt.param_groups: g['lr'] = lr_now
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(use_amp and X.is_cuda))
        with ctx:
            yhat = model(X[j]); loss = F.mse_loss(yhat, y[j])
            if extra_loss_fn is not None: loss = loss + extra_loss_fn()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); steps_done += 1
    if X.is_cuda:
        end.record(); torch.cuda.synchronize()
        ms = start.elapsed_time(end) / max(1, (n + batch - 1)//batch)
    else:
        ms = (time.perf_counter() - t0) * 1000.0 / max(1, (n + batch - 1)//batch)
    return steps_done, ms

@torch.no_grad()
def eval_mse(model, X, y): model.eval(); return F.mse_loss(model(X), y).item()

@torch.no_grad()
def eval_mse_val(model, Xva, yva): model.eval(); return F.mse_loss(model(Xva), yva).item()

@torch.no_grad()
def warmup_model(model, X, steps=2, batch=64):
    if not X.is_cuda: return
    model.train(); n = min(4*batch, X.size(0)); perm = torch.randperm(X.size(0), device=X.device)[:n]
    for _ in range(steps): _ = model(X[perm[:batch]])
    torch.cuda.synchronize()

def train_with_early_stop(model, opt, Xtr, ytr, Xva, yva, batch, base_lr, steps_total,
                          use_amp=True, max_epochs=120, patience=EARLY_STOP_PATIENCE,
                          extra_loss_fn=None) -> Tuple[float, int]:
    best=float('inf'); best_state=None; ms_list=[]; steps=0; wait=0
    for ep in range(max_epochs):
        steps, ms = train_epoch(model, opt, Xtr, ytr, batch, base_lr, steps, steps_total, use_amp, extra_loss_fn)
        ms_list.append(ms); val = eval_mse_val(model, Xva, yva)
        if val + 1e-8 < best: best=val; wait=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: wait += 1
        if wait >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    avg_ms = sum(ms_list[1:]) / max(1, len(ms_list)-1)
    return avg_ms, (ep+1)

def param_parity_M(H:int) -> int:
    return max(1, 3*H - 5)

# ===== Runner =====
def run_experiment(
    name:str,
    model_type:str,          # "baseline" | "sine" | "wave" | "fwrff"
    H:int,
    M: Optional[int] = None,
    epochs:int=120,
    batch:int=256,
    seeds: List[int] = [1337],
    lr_main: float = 2.0e-2,
    lr_fwrff: float = 1.0e-2,          # a bit higher helps hit lower MSE
    fwrff_octaves:int = 2,
    fwrff_ratio: float = 1.7,
    fwrff_gate_l1: float = 1e-4,
    fwrff_hidden_act: str = "sine",
    fwrff_warp: bool = True,
    fwrff_rank:int = 4,
    fwrff_phi_fp16: bool = True,
    amp_override: Optional[bool] = None,
    n_data:int = 900,
    save_csv: Optional[str] = None,
):
    results = []
    for seed in seeds:
        Xtr,ytr,Xva,yva,Xte,yte = make_data(n=n_data, seed=seed, device=DEVICE, dtype=DTYPE)
        steps_total = epochs * math.ceil(Xtr.size(0) / batch)
        use_amp = USE_AMP if amp_override is None else amp_override

        if model_type == "baseline":
            model = BaselineMLP(H).to(DEVICE); model = maybe_compile(model)
            warmup_model(model, Xtr)
            opt = torch.optim.AdamW(model.parameters(), lr=lr_main, weight_decay=1e-4)
            ms, used = train_with_early_stop(model, opt, Xtr, ytr, Xva, yva, batch, lr_main, steps_total, use_amp, epochs)
            test = eval_mse(model, Xte, yte)
            results.append({"exp":name,"seed":seed,"model":"Baseline","H":H,"M":"-","epochs_used":used,
                            "batch":batch,"ms_per_epoch":ms,"test_mse":test})

        elif model_type == "sine":
            model = SineMLP(H).to(DEVICE); model = maybe_compile(model)
            warmup_model(model, Xtr)
            opt = torch.optim.AdamW(model.parameters(), lr=lr_main, weight_decay=1e-4)
            ms, used = train_with_early_stop(model, opt, Xtr, ytr, Xva, yva, batch, lr_main, steps_total, use_amp, epochs)
            test = eval_mse(model, Xte, yte)
            results.append({"exp":name,"seed":seed,"model":"SineMLP","H":H,"M":"-","epochs_used":used,
                            "batch":batch,"ms_per_epoch":ms,"test_mse":test})

        elif model_type == "wave":
            model = WaveBasisMLP(H).to(DEVICE); model = maybe_compile(model)
            warmup_model(model, Xtr)
            opt = torch.optim.AdamW(model.parameters(), lr=lr_main, weight_decay=1e-4)
            ms, used = train_with_early_stop(model, opt, Xtr, ytr, Xva, yva, batch, lr_main, steps_total, use_amp, epochs)
            test = eval_mse(model, Xte, yte)
            results.append({"exp":name,"seed":seed,"model":"WaveBasisMLP","H":H,"M":"-","epochs_used":used,
                            "batch":batch,"ms_per_epoch":ms,"test_mse":test})

        elif model_type == "fwrff":
            M_eff = param_parity_M(H) if (M is None) else M
            cfg = FWRFFConfig(H=H, M=M_eff, octaves=fwrff_octaves, octave_ratio=fwrff_ratio,
                              gate_l1=fwrff_gate_l1, hidden_act=fwrff_hidden_act, dtype=DTYPE,
                              warp_enabled=fwrff_warp, rank_residual=fwrff_rank,
                              phi_store_fp16=fwrff_phi_fp16)
            model = FractalWaveRFF(cfg, device=DEVICE).to(DEVICE); model = maybe_compile(model)
            warmup_model(model, Xtr)
            opt = torch.optim.AdamW(model.parameters(), lr=lr_fwrff, weight_decay=1e-4, betas=(0.9, 0.98))
            ms, used = train_with_early_stop(
                model, opt, Xtr, ytr, Xva, yva, batch, lr_fwrff, steps_total,
                use_amp, epochs, extra_loss_fn=model.gate_l1_loss
            )
            test = eval_mse(model, Xte, yte)
            results.append({"exp":name,"seed":seed,"model":f"FractalWaveRFF(O={cfg.octaves},{cfg.hidden_act},warp={cfg.warp_enabled},rank={cfg.rank_residual},phi16={cfg.phi_store_fp16})",
                            "H":H,"M":M_eff,"epochs_used":used,"batch":batch,
                            "ms_per_epoch":ms,"test_mse":test})
        else:
            raise ValueError(f"Unknown model_type {model_type}")

    print(f"\n=== Results: {name} ===")
    hdr = ["exp","seed","model","H","M","epochs_used","batch","ms_per_epoch","test_mse"]
    print("\t".join(hdr))
    for r in results: print("\t".join(str(r[k]) for k in hdr))
    if save_csv:
        with open(save_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if f.tell() == 0: w.writeheader()
            for r in results: w.writerow(r)
    return results

# ===== Experiments =====
EXPERIMENTS: List[Dict] = [
    # Baselines @ H=8
    dict(name="H8_baseline", model_type="baseline", H=8, epochs=120, batch=256, seeds=[101,133,202]),
    dict(name="H8_sine",     model_type="sine",     H=8, epochs=120, batch=256, seeds=[101,133,202]),
    dict(name="H8_wave",     model_type="wave",     H=8, epochs=120, batch=256, seeds=[101,133,202]),

    # FWRFF parity O=2 vs O=3 (warp ON, rank=4, phi fp16) — AMP OFF
    dict(name="H8_fwrff_parity_O2_rank4", model_type="fwrff", H=8, M=None, epochs=120, batch=512,
         seeds=[101,133,202], fwrff_octaves=2, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=True,
         amp_override=False),
    dict(name="H8_fwrff_parity_O3_rank4", model_type="fwrff", H=8, M=None, epochs=120, batch=512,
         seeds=[101,133,202], fwrff_octaves=3, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=True,
         amp_override=False),

    # M up: 27 and 35 (O=2)
    dict(name="H8_fwrff_M27_O2", model_type="fwrff", H=8, M=27, epochs=120, batch=512,
         seeds=[101,133,202], fwrff_octaves=2, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=True,
         amp_override=False),
    dict(name="H8_fwrff_M35_O2", model_type="fwrff", H=8, M=35, epochs=120, batch=512,
         seeds=[101,133,202], fwrff_octaves=2, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=True,
         amp_override=False),

    # A/B: Φ fp32 storage
    dict(name="H8_fwrff_parity_O2_phi32", model_type="fwrff", H=8, M=None, epochs=120, batch=512,
         seeds=[101,133,202], fwrff_octaves=2, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=False,
         amp_override=False),

    # H=16 reference + parity FWRFF
    dict(name="H16_baseline", model_type="baseline", H=16, epochs=120, batch=256, seeds=[202,303]),
    dict(name="H16_sine",     model_type="sine",     H=16, epochs=120, batch=256, seeds=[202,303]),
    dict(name="H16_wave",     model_type="wave",     H=16, epochs=120, batch=256, seeds=[202,303]),
    dict(name="H16_fwrff_parity_O2_rank4", model_type="fwrff", H=16, M=None, epochs=120, batch=512,
         seeds=[202,303], fwrff_octaves=2, fwrff_ratio=1.7, lr_fwrff=1e-2,
         fwrff_hidden_act="sine", fwrff_warp=True, fwrff_rank=4, fwrff_phi_fp16=True,
         amp_override=False),
]

def main():
    for cfg in EXPERIMENTS:
        run_experiment(**cfg)

if __name__ == "__main__":
    if DEVICE == "cuda": torch.cuda.empty_cache()
    main()
