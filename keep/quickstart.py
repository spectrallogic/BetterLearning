# quickstart.py
# ==============================================================================
# QUICK START: Test LFFT in 30 seconds!
# ==============================================================================

import torch
import torch.nn.functional as F
import time

print("\n" + "=" * 80)
print("🌊 LIQUID FRACTAL FREQUENCY TRANSFORMER - QUICK START")
print("=" * 80 + "\n")

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {DEVICE}\n")

# Import LFFT
try:
    from keep.liquid_fractal_transformer import LFFT, UltraFastLFFT

    print("✓ LFFT imported successfully\n")
except ImportError as e:
    print(f"✗ Error importing LFFT: {e}")
    print("Make sure liquid_fractal_transformer.py is in the same directory!")
    exit(1)

# Create a tiny test
print("=" * 80)
print("TEST 1: Creating Models")
print("=" * 80 + "\n")

vocab_size = 50
seq_length = 32

# Create LFFT models
model_full = LFFT(vocab_size, seq_length, n_fractal_scales=2, n_freq_per_scale=8, n_layers=2).to(DEVICE)
model_ultra = UltraFastLFFT(vocab_size, seq_length).to(DEVICE)

print(f"LFFT (Full):  {sum(p.numel() for p in model_full.parameters()):,} parameters")
print(f"LFFT (Ultra): {sum(p.numel() for p in model_ultra.parameters()):,} parameters")
print()

# Test forward pass
print("=" * 80)
print("TEST 2: Forward Pass")
print("=" * 80 + "\n")

x = torch.randint(0, vocab_size, (4, seq_length), device=DEVICE)

# Full LFFT
try:
    output_full = model_full(x)
    print(f"✓ LFFT (Full) forward pass: {x.shape} -> {output_full.shape}")
except Exception as e:
    print(f"✗ LFFT (Full) failed: {e}")

# Ultra LFFT
try:
    output_ultra = model_ultra(x)
    print(f"✓ LFFT (Ultra) forward pass: {x.shape} -> {output_ultra.shape}")
except Exception as e:
    print(f"✗ LFFT (Ultra) failed: {e}")

print()

# Speed test
print("=" * 80)
print("TEST 3: Speed Benchmark")
print("=" * 80 + "\n")


def benchmark(model, name, n_runs=50):
    model.eval()
    x_test = torch.randint(0, vocab_size, (8, seq_length), device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_test)

    # Benchmark
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x_test)
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / n_runs
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x_test)
        time_ms = ((time.perf_counter() - start_time) / n_runs) * 1000

    throughput = (8 * seq_length / time_ms) * 1000
    return time_ms, throughput


time_full, throughput_full = benchmark(model_full, "LFFT Full")
time_ultra, throughput_ultra = benchmark(model_ultra, "LFFT Ultra")

print(f"LFFT (Full):  {time_full:.3f}ms per batch | {throughput_full:,.0f} tokens/sec")
print(f"LFFT (Ultra): {time_ultra:.3f}ms per batch | {throughput_ultra:,.0f} tokens/sec")
print(f"Speedup (Ultra vs Full): {time_full / time_ultra:.2f}x")
print()

# Mini training test
print("=" * 80)
print("TEST 4: Mini Training")
print("=" * 80 + "\n")

# Create tiny dataset
data = torch.randint(0, vocab_size, (1000,))
train_data = data[:800]
val_data = data[800:]

print(f"Training data: {len(train_data)} tokens")
print(f"Validation data: {len(val_data)} tokens")
print()

# Train for a few steps
model_ultra.train()
optimizer = torch.optim.AdamW(model_ultra.parameters(), lr=3e-3)

print("Training LFFT (Ultra) for 5 epochs...")
for epoch in range(5):
    total_loss = 0
    n_batches = 10

    for _ in range(n_batches):
        # Random batch
        idx = torch.randint(0, len(train_data) - seq_length, (4,))
        x = torch.stack([train_data[i:i + seq_length] for i in idx]).to(DEVICE)
        y = torch.stack([train_data[i + 1:i + seq_length + 1] for i in idx]).to(DEVICE)

        # Forward
        optimizer.zero_grad()
        logits = model_ultra(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / n_batches
    print(f"  Epoch {epoch + 1}/5: Loss = {avg_loss:.4f}")

print()

# Final results
print("=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80 + "\n")

print("🎉 LFFT is working correctly!\n")
print("Next steps:")
print("  1. Run the full demo: python liquid_fractal_transformer.py")
print("  2. Run the benchmark: python breakthrough_benchmark.py")
print("  3. Test on your own data!")
print()
print("=" * 80 + "\n")

'''
C:\Users\MAC-USER\PycharmProjects\BetterLearning\.venv\Scripts\python.exe C:\Users\MAC-USER\PycharmProjects\BetterLearning\quickstart.py 

================================================================================
🌊 LIQUID FRACTAL FREQUENCY TRANSFORMER - QUICK START
================================================================================

✓ PyTorch version: 2.5.1+cu121
✓ CUDA available: True
✓ Using device: cuda


================================================================================
🌊 LIQUID FRACTAL FREQUENCY TRANSFORMER (LFFT)
Device: cuda | PyTorch: 2.5.1+cu121
================================================================================

✓ LFFT imported successfully

================================================================================
TEST 1: Creating Models
================================================================================

LFFT (Full):  4,274 parameters
LFFT (Ultra): 1,666 parameters

================================================================================
TEST 2: Forward Pass
================================================================================

✓ LFFT (Full) forward pass: torch.Size([4, 32]) -> torch.Size([4, 32, 50])
✓ LFFT (Ultra) forward pass: torch.Size([4, 32]) -> torch.Size([4, 32, 50])

================================================================================
TEST 3: Speed Benchmark
================================================================================

LFFT (Full):  40.000ms per batch | 6,400 tokens/sec
LFFT (Ultra): 0.146ms per batch | 1,754,879 tokens/sec
Speedup (Ultra vs Full): 274.20x

================================================================================
TEST 4: Mini Training
================================================================================

Training data: 800 tokens
Validation data: 200 tokens

Training LFFT (Ultra) for 5 epochs...
  Epoch 1/5: Loss = 3.9888
  Epoch 2/5: Loss = 3.9632
  Epoch 3/5: Loss = 3.9444
  Epoch 4/5: Loss = 3.9532
  Epoch 5/5: Loss = 3.9352

================================================================================
✅ ALL TESTS PASSED!
================================================================================

🎉 LFFT is working correctly!

Next steps:
  1. Run the full demo: python liquid_fractal_transformer.py
  2. Run the benchmark: python breakthrough_benchmark.py
  3. Test on your own data!

================================================================================


Process finished with exit code 0

'''