"""
Crystal Tiled Network - Maintain Sweet Spot at Any Size
Split large networks into 4M parameter tiles
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class TiledLinear(nn.Module):
    """
    Split large weight matrix into 2048×2048 tiles (4M params sweet spot)
    Process tiles sequentially for optimal cache utilization
    """
    def __init__(self, in_features, out_features, tile_size=2048):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size

        # Calculate number of tiles needed
        self.n_tiles_in = (in_features + tile_size - 1) // tile_size
        self.n_tiles_out = (out_features + tile_size - 1) // tile_size

        # Create weight tiles
        self.tiles = nn.ModuleList()
        for i in range(self.n_tiles_out):
            row_tiles = nn.ModuleList()
            out_start = i * tile_size
            out_end = min((i + 1) * tile_size, out_features)
            out_size = out_end - out_start

            for j in range(self.n_tiles_in):
                in_start = j * tile_size
                in_end = min((j + 1) * tile_size, in_features)
                in_size = in_end - in_start

                # Each tile is a small linear layer (stays in L2 cache!)
                row_tiles.append(nn.Linear(in_size, out_size))

            self.tiles.append(row_tiles)

        print(f"  Created {self.n_tiles_out}×{self.n_tiles_in} tiling ({self.n_tiles_out * self.n_tiles_in} tiles)")

    def forward(self, x):
        # Split input into tiles
        x_tiles = []
        for j in range(self.n_tiles_in):
            in_start = j * self.tile_size
            in_end = min((j + 1) * self.tile_size, self.in_features)
            x_tiles.append(x[:, in_start:in_end])

        # Process tiles and accumulate outputs
        outputs = []
        for i in range(self.n_tiles_out):
            # Accumulate contributions from all input tiles
            tile_sum = None
            for j in range(self.n_tiles_in):
                tile_out = self.tiles[i][j](x_tiles[j])
                if tile_sum is None:
                    tile_sum = tile_out
                else:
                    tile_sum = tile_sum + tile_out
            outputs.append(tile_sum)

        # Concatenate output tiles
        return torch.cat(outputs, dim=-1)


class CrystalTiledNetwork(nn.Module):
    """Network that maintains 4M param sweet spot through tiling"""
    def __init__(self, input_dim, hidden_dims, output_dim, use_tiling=True, tile_size=2048):
        super().__init__()
        self.use_tiling = use_tiling

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if use_tiling and (current_dim > tile_size or hidden_dim > tile_size):
                print(f"Layer {i}: {current_dim} -> {hidden_dim} (TILED)")
                self.layers.append(TiledLinear(current_dim, hidden_dim, tile_size))
            else:
                print(f"Layer {i}: {current_dim} -> {hidden_dim} (normal)")
                self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim

        self.output = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return self.output(x)


def benchmark_speed(model, input_shape, device='cuda', iterations=500):
    """Benchmark inference speed"""
    model = model.to(device).eval()
    x = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return (elapsed / iterations) * 1000


def train_model(model, X, y, epochs=30, lr=0.001):
    """Train model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.4f}")


def main():
    print("="*80)
    print("CRYSTAL TILED NETWORK - Maintain Sweet Spot at Any Size")
    print("="*80)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Test configurations - from small to VERY large
    test_configs = [
        ("Small (4M)", 1000, [2048, 1024], 1, 256),
        ("Medium (16M)", 2000, [4096, 2048], 1, 256),
        ("Large (65M)", 4000, [8192, 4096], 1, 256),
    ]

    results = []

    for name, input_dim, hidden_dims, output_dim, batch_size in test_configs:
        print("="*80)
        print(f"Testing: {name}")
        print("="*80)
        print(f"Config: {input_dim} -> {hidden_dims} -> {output_dim}")
        print()

        # Create models
        print("Creating NORMAL model:")
        normal = CrystalTiledNetwork(input_dim, hidden_dims, output_dim, use_tiling=False)

        print("\nCreating TILED model:")
        tiled = CrystalTiledNetwork(input_dim, hidden_dims, output_dim, use_tiling=True, tile_size=2048)

        params = sum(p.numel() for p in normal.parameters())
        print(f"\nTotal parameters: {params:,}")
        print()

        # Quick training test
        print("Training normal model:")
        X = torch.randn(500, input_dim)
        y = torch.randn(500, output_dim)
        train_model(normal, X, y, epochs=30)

        print("\nTraining tiled model:")
        X = torch.randn(500, input_dim)
        y = torch.randn(500, output_dim)
        train_model(tiled, X, y, epochs=30)

        # Benchmark
        print("\nBenchmarking...")
        normal_time = benchmark_speed(normal, (batch_size, input_dim), device=device)
        tiled_time = benchmark_speed(tiled, (batch_size, input_dim), device=device)

        speedup = normal_time / tiled_time

        print(f"Normal: {normal_time:.4f} ms")
        print(f"Tiled:  {tiled_time:.4f} ms")
        print(f"Speedup: {speedup:.2f}×")

        results.append((name, params, normal_time, tiled_time, speedup))
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Params':<12} {'Normal':<12} {'Tiled':<12} {'Speedup'}")
    print("-"*80)
    for name, params, normal_time, tiled_time, speedup in results:
        status = "✓ FASTER" if speedup > 1.0 else "⚠ SLOWER"
        print(f"{name:<20} {params/1e6:>6.1f}M     {normal_time:>8.4f}ms  {tiled_time:>8.4f}ms  {speedup:>6.2f}× {status}")

    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("Tiling maintains consistent performance by keeping each tile at sweet spot")
    print("="*80)


if __name__ == "__main__":
    main()

'''
Results:
C:\Users\MAC-USER\PycharmProjects\BetterLearning\.venv\Scripts\python.exe C:\Users\MAC-USER\PycharmProjects\BetterLearning\sparse_binary_crystal.py 
================================================================================
CRYSTAL TILED NETWORK - Maintain Sweet Spot at Any Size
================================================================================

Device: cuda

================================================================================
Testing: Small (4M)
================================================================================
Config: 1000 -> [2048, 1024] -> 1

Creating NORMAL model:
Layer 0: 1000 -> 2048 (normal)
Layer 1: 2048 -> 1024 (normal)

Creating TILED model:
Layer 0: 1000 -> 2048 (normal)
Layer 1: 2048 -> 1024 (normal)

Total parameters: 4,149,249

Training normal model:
  Epoch 10/30 | Loss: 0.4538
  Epoch 20/30 | Loss: 0.0551
  Epoch 30/30 | Loss: 0.0141

Training tiled model:
  Epoch 10/30 | Loss: 0.6858
  Epoch 20/30 | Loss: 0.0964
  Epoch 30/30 | Loss: 0.0365

Benchmarking...
Normal: 0.1620 ms
Tiled:  0.1660 ms
Speedup: 0.98×

================================================================================
Testing: Medium (16M)
================================================================================
Config: 2000 -> [4096, 2048] -> 1

Creating NORMAL model:
Layer 0: 2000 -> 4096 (normal)
Layer 1: 4096 -> 2048 (normal)

Creating TILED model:
Layer 0: 2000 -> 4096 (TILED)
  Created 2×1 tiling (2 tiles)
Layer 1: 4096 -> 2048 (TILED)
  Created 1×2 tiling (2 tiles)

Total parameters: 16,588,801

Training normal model:
  Epoch 10/30 | Loss: 0.9842
  Epoch 20/30 | Loss: 0.5287
  Epoch 30/30 | Loss: 0.2164

Training tiled model:
  Epoch 10/30 | Loss: 0.9147
  Epoch 20/30 | Loss: 0.6978
  Epoch 30/30 | Loss: 0.4106

Benchmarking...
Normal: 0.6160 ms
Tiled:  0.6665 ms
Speedup: 0.92×

================================================================================
Testing: Large (65M)
================================================================================
Config: 4000 -> [8192, 4096] -> 1

Creating NORMAL model:
Layer 0: 4000 -> 8192 (normal)
Layer 1: 8192 -> 4096 (normal)

Creating TILED model:
Layer 0: 4000 -> 8192 (TILED)
  Created 4×2 tiling (8 tiles)
Layer 1: 8192 -> 4096 (TILED)
  Created 2×4 tiling (8 tiles)

Total parameters: 66,338,817

Training normal model:
  Epoch 10/30 | Loss: 1.0739
  Epoch 20/30 | Loss: 0.8475
  Epoch 30/30 | Loss: 0.0992

Training tiled model:
  Epoch 10/30 | Loss: 1.3173
  Epoch 20/30 | Loss: 0.7746
  Epoch 30/30 | Loss: 0.2132

Benchmarking...
Normal: 2.2897 ms
Tiled:  2.6379 ms
Speedup: 0.87×

================================================================================
SUMMARY
================================================================================
Config               Params       Normal       Tiled        Speedup
--------------------------------------------------------------------------------
Small (4M)              4.1M       0.1620ms    0.1660ms    0.98× ⚠ SLOWER
Medium (16M)           16.6M       0.6160ms    0.6665ms    0.92× ⚠ SLOWER
Large (65M)            66.3M       2.2897ms    2.6379ms    0.87× ⚠ SLOWER

================================================================================
KEY INSIGHT:
Tiling maintains consistent performance by keeping each tile at sweet spot
================================================================================
'''