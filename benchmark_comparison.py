"""
ExpandFormer Benchmark Suite
=============================

Compares your ExpandFormer architecture against baseline transformers to answer:
1. Does it learn faster?
2. Is it more parameter-efficient?
3. Does real-time learning work better?
4. Does growth help performance?
5. Is the two-speed approach effective?

USAGE:
python benchmark_comparison.py --quick        # Fast comparison (~5 min)
python benchmark_comparison.py --full         # Full comparison (~30 min)
python benchmark_comparison.py --realtime     # Test real-time learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

try:
    import tiktoken
except ImportError:
    print("ERROR: pip install tiktoken")
    sys.exit(1)


# ============================================================================
# BASELINE: Standard Transformer (for comparison)
# ============================================================================

class BaselineTransformer(nn.Module):
    """
    Standard transformer - no tricks, no growth
    This is what we're comparing against
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 context_len=256, num_layers=2, num_heads=2, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # Standard components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._create_pos_encoding(context_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Standard transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def _create_pos_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, x):
        batch_size, seq_len = x.shape

        h = self.embedding(x)
        h = self.embed_dropout(h)

        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        h = self.input_proj(h)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.output_norm(h)
        logits = self.output(h)

        return logits


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

@dataclass
class BenchmarkMetrics:
    """Metrics for comparing models"""
    model_name: str

    # Performance
    final_loss: float = 0.0
    final_perplexity: float = 0.0
    best_loss: float = float('inf')

    # Efficiency
    total_params: int = 0
    active_params: int = 0  # For growing models
    training_time: float = 0.0

    # Learning dynamics
    loss_history: list = None
    perplexity_history: list = None

    # Real-time adaptation
    realtime_adaptation_score: float = 0.0

    # Model-specific
    num_splits: int = 0  # For ExpandFormer
    fast_confidence: float = 0.0  # For two-speed models

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.perplexity_history is None:
            self.perplexity_history = []

    def to_dict(self):
        return {
            'model_name': self.model_name,
            'final_loss': self.final_loss,
            'final_perplexity': self.final_perplexity,
            'best_loss': self.best_loss,
            'total_params': self.total_params,
            'active_params': self.active_params,
            'training_time': self.training_time,
            'realtime_adaptation_score': self.realtime_adaptation_score,
            'num_splits': self.num_splits,
            'param_efficiency': self.best_loss / (self.active_params / 1e6) if self.active_params > 0 else 0,
        }


class BenchmarkEvaluator:
    """Evaluates and compares models"""

    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}

    def compute_perplexity(self, loss):
        """Convert loss to perplexity"""
        return math.exp(min(loss, 20))  # Cap to prevent overflow

    def evaluate_on_dataset(self, model, texts, context_len=128):
        """Compute loss and perplexity on dataset"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                for i in range(1, len(tokens)):
                    context_start = max(0, i - context_len)
                    context = tokens[context_start:i]

                    if len(context) < context_len:
                        context = [0] * (context_len - len(context)) + context

                    x = torch.tensor([context[-context_len:]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    logits = model(x)
                    loss = F.cross_entropy(logits[:, -1, :], y)

                    total_loss += loss.item()
                    total_tokens += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = self.compute_perplexity(avg_loss)

        model.train()
        return avg_loss, perplexity

    def test_realtime_adaptation(self, model, optimizer, test_phrases, context_len=128):
        """
        Test how quickly model adapts to new patterns

        Give it new phrases and see how fast loss decreases
        """
        model.train()
        adaptation_losses = []

        for phrase in test_phrases:
            tokens = self.tokenizer.encode(phrase)
            if len(tokens) < 2:
                continue

            phrase_losses = []

            # Train on this phrase multiple times
            for iteration in range(10):
                for i in range(1, len(tokens)):
                    context_start = max(0, i - context_len)
                    context = tokens[context_start:i]

                    if len(context) < context_len:
                        context = [0] * (context_len - len(context)) + context

                    x = torch.tensor([context[-context_len:]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits[:, -1, :], y)
                    loss.backward()
                    optimizer.step()

                    phrase_losses.append(loss.item())

            # How fast did it learn? (compare first vs last)
            if len(phrase_losses) >= 2:
                improvement = phrase_losses[0] - phrase_losses[-1]
                adaptation_losses.append(improvement)

        # Average improvement across phrases
        return np.mean(adaptation_losses) if adaptation_losses else 0.0


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Runs comprehensive benchmarks"""

    def __init__(self, training_texts, test_texts, device='cuda'):
        self.training_texts = training_texts
        self.test_texts = test_texts
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.evaluator = BenchmarkEvaluator(self.tokenizer, device)

    def train_model(self, model, model_name, num_updates=1000, context_len=128):
        """Train a model and track metrics"""
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}\n")

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

        metrics = BenchmarkMetrics(model_name=model_name)
        metrics.total_params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        update_count = 0
        recent_losses = deque(maxlen=100)

        print(f"Parameters: {metrics.total_params:,}")
        print(f"Target updates: {num_updates:,}\n")

        # Training loop
        while update_count < num_updates:
            for text in self.training_texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                for i in range(1, len(tokens)):
                    context_start = max(0, i - context_len)
                    context = tokens[context_start:i]

                    if len(context) < context_len:
                        context = [0] * (context_len - len(context)) + context

                    x = torch.tensor([context[-context_len:]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits[:, -1, :], y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    recent_losses.append(loss.item())
                    update_count += 1

                    # Track metrics every 100 updates
                    if update_count % 100 == 0:
                        avg_loss = np.mean(list(recent_losses))
                        perplexity = self.evaluator.compute_perplexity(avg_loss)

                        metrics.loss_history.append(avg_loss)
                        metrics.perplexity_history.append(perplexity)
                        metrics.best_loss = min(metrics.best_loss, avg_loss)

                        print(f"Update {update_count:5d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f}")

                    if update_count >= num_updates:
                        break

                if update_count >= num_updates:
                    break

        metrics.training_time = time.time() - start_time

        # Final evaluation on test set
        test_loss, test_ppl = self.evaluator.evaluate_on_dataset(model, self.test_texts, context_len)
        metrics.final_loss = test_loss
        metrics.final_perplexity = test_ppl
        metrics.active_params = metrics.total_params

        print(f"\n✓ Training complete!")
        print(f"  Final test loss: {test_loss:.4f}")
        print(f"  Final test perplexity: {test_ppl:.2f}")
        print(f"  Training time: {metrics.training_time:.1f}s")

        return metrics

    def train_expandformer(self, model, model_name, num_updates=1000, context_len=128):
        """Train ExpandFormer with growth tracking"""
        print(f"\n{'='*70}")
        print(f"Training: {model_name} (with growth)")
        print(f"{'='*70}\n")

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

        metrics = BenchmarkMetrics(model_name=model_name)
        metrics.total_params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        update_count = 0
        recent_losses = deque(maxlen=100)

        print(f"Initial parameters: {metrics.total_params:,}")
        print(f"Target updates: {num_updates:,}\n")

        # Training loop
        while update_count < num_updates:
            for text in self.training_texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                for i in range(1, len(tokens)):
                    context_start = max(0, i - context_len)
                    context = tokens[context_start:i]

                    if len(context) < context_len:
                        context = [0] * (context_len - len(context)) + context

                    x = torch.tensor([context[-context_len:]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits[:, -1, :], y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    recent_losses.append(loss.item())
                    update_count += 1

                    # Update block difficulties (if ExpandFormer)
                    if hasattr(model, 'get_all_blocks'):
                        for block in model.get_all_blocks():
                            block.update_difficulty(loss.item(), x)

                    # Check for growth
                    if hasattr(model, 'check_and_split') and update_count % 50 == 0:
                        avg_loss = np.mean(list(recent_losses))
                        if model.check_and_split(avg_loss):
                            metrics.num_splits += 1
                            new_params = sum(p.numel() for p in model.parameters())
                            print(f"  🌱 Split! Now {new_params:,} params")

                    # Track metrics every 100 updates
                    if update_count % 100 == 0:
                        avg_loss = np.mean(list(recent_losses))
                        perplexity = self.evaluator.compute_perplexity(avg_loss)

                        metrics.loss_history.append(avg_loss)
                        metrics.perplexity_history.append(perplexity)
                        metrics.best_loss = min(metrics.best_loss, avg_loss)

                        active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                        print(f"Update {update_count:5d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Params: {active_params:,}")

                    if update_count >= num_updates:
                        break

                if update_count >= num_updates:
                    break

        metrics.training_time = time.time() - start_time

        # Final stats
        metrics.active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(model, 'fast_confidence'):
            metrics.fast_confidence = model.fast_confidence

        # Final evaluation on test set
        test_loss, test_ppl = self.evaluator.evaluate_on_dataset(model, self.test_texts, context_len)
        metrics.final_loss = test_loss
        metrics.final_perplexity = test_ppl

        print(f"\n✓ Training complete!")
        print(f"  Final test loss: {test_loss:.4f}")
        print(f"  Final test perplexity: {test_ppl:.2f}")
        print(f"  Training time: {metrics.training_time:.1f}s")
        print(f"  Splits: {metrics.num_splits}")
        print(f"  Final params: {metrics.active_params:,}")

        return metrics

    def compare_realtime_learning(self, models_and_names, test_phrases):
        """Compare real-time adaptation capabilities"""
        print(f"\n{'='*70}")
        print("REAL-TIME ADAPTATION TEST")
        print(f"{'='*70}\n")

        results = {}

        for model, name in models_and_names:
            print(f"\nTesting: {name}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

            score = self.evaluator.test_realtime_adaptation(
                model, optimizer, test_phrases
            )

            results[name] = score
            print(f"  Adaptation score: {score:.4f}")

        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(metrics_list, save_path='benchmark_results'):
    """Create comparison plots"""
    Path(save_path).mkdir(exist_ok=True)

    # 1. Loss curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for metrics in metrics_list:
        plt.plot(metrics.loss_history, label=metrics.model_name, linewidth=2)
    plt.xlabel('Update (x100)')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for metrics in metrics_list:
        plt.plot(metrics.perplexity_history, label=metrics.model_name, linewidth=2)
    plt.xlabel('Update (x100)')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/loss_curves.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {save_path}/loss_curves.png")
    plt.close()

    # 2. Parameter efficiency
    plt.figure(figsize=(10, 6))

    names = [m.model_name for m in metrics_list]
    final_losses = [m.final_loss for m in metrics_list]
    params = [m.active_params / 1e6 for m in metrics_list]  # In millions

    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_list)))

    for i, (name, loss, param, color) in enumerate(zip(names, final_losses, params, colors)):
        plt.scatter(param, loss, s=200, c=[color], label=name, alpha=0.7, edgecolors='black')

    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Final Test Loss')
    plt.title('Parameter Efficiency (Lower-Left is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/param_efficiency.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {save_path}/param_efficiency.png")
    plt.close()

    # 3. Summary table
    fig, ax = plt.subplots(figsize=(12, len(metrics_list) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    headers = ['Model', 'Test Loss', 'Test PPL', 'Params (M)', 'Time (s)', 'Best Loss', 'Splits']

    for m in metrics_list:
        row = [
            m.model_name,
            f"{m.final_loss:.4f}",
            f"{m.final_perplexity:.2f}",
            f"{m.active_params/1e6:.2f}",
            f"{m.training_time:.1f}",
            f"{m.best_loss:.4f}",
            f"{m.num_splits}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{save_path}/summary_table.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {save_path}/summary_table.png")
    plt.close()


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(mode='quick'):
    """Run full benchmark suite"""
    print("="*70)
    print("🏁 EXPANDFORMER BENCHMARK SUITE")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Mode: {mode}\n")

    # Load data
    training_dir = Path("training_data")
    all_texts = []

    if training_dir.exists():
        print("📂 Loading data...")
        for file_path in training_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                lines = [line.strip() for line in text.split('\n') if line.strip()]

                for i in range(0, len(lines), 6):
                    chunk = '\n'.join(lines[i:i+6])
                    if len(chunk) > 10:
                        all_texts.append(chunk)

    if not all_texts:
        print("⚠️  Using demo data")
        all_texts = [
            "Hello, how are you? I am doing well, thank you.",
            "The weather is nice today. The sky is blue.",
            "What is your favorite color? I like blue and green.",
            "Learning new things is always exciting and fun.",
            "I enjoy reading books about science and history.",
        ] * 10

    # Split train/test
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    test_texts = all_texts[split_idx:]

    print(f"✓ Train: {len(train_texts)} chunks")
    print(f"✓ Test: {len(test_texts)} chunks\n")

    # Set parameters based on mode
    if mode == 'quick':
        num_updates = 500
        context_len = 64
    else:
        num_updates = 2000
        context_len = 128

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # Create benchmark runner
    runner = BenchmarkRunner(train_texts, test_texts, device=device)

    all_metrics = []

    # 1. Baseline Transformer (Small)
    print("\n" + "="*70)
    print("MODEL 1: Baseline Transformer (Small)")
    print("="*70)
    baseline_small = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=2
    )
    metrics_baseline_small = runner.train_model(
        baseline_small, "Baseline (Small)", num_updates, context_len
    )
    all_metrics.append(metrics_baseline_small)

    # 2. Baseline Transformer (Large - same as ExpandFormer final size)
    print("\n" + "="*70)
    print("MODEL 2: Baseline Transformer (Large)")
    print("="*70)
    baseline_large = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=4,  # More layers
        num_heads=2
    )
    metrics_baseline_large = runner.train_model(
        baseline_large, "Baseline (Large)", num_updates, context_len
    )
    all_metrics.append(metrics_baseline_large)

    # 3. ExpandFormer v11 (Your FIXED Architecture)
    try:
        # Import your FIXED architecture
        sys.path.insert(0, str(Path.cwd()))

        # Try v11 first (the FIXED version)
        try:
            from expandformer_v11 import EfficientTwoSpeedTransformer as TwoSpeedTransformer
            version_name = "v11 (FIXED)"
            use_max_splits = True
        except ImportError:
            # Fall back to v9/v10 if v11 not available
            try:
                from expandformer_v9 import TwoSpeedTransformer
                version_name = "v9 (OLD)"
                use_max_splits = False
            except ImportError:
                from expandformer_v10 import TwoSpeedTransformer
                version_name = "v10 (OLD)"
                use_max_splits = False

        print("\n" + "="*70)
        print(f"MODEL 3: ExpandFormer {version_name}")
        print("="*70)
        print("Features:")
        print("  • Two-speed learning (fast associations + slow understanding)")
        print("  • Adaptive growth (splits when stuck)")
        print("  • Difficulty-aware (tracks what it struggles with)")
        print("  • Adaptive loss weighting (values easy learning)")
        if use_max_splits:
            print("  • FIXED: Conservative splitting with max_splits=3")
            print("  • FIXED: Starts 25% smaller for efficiency\n")
        else:
            print("  ⚠️  WARNING: Using old version (will over-split)\n")

        # Create model with appropriate parameters
        if use_max_splits:
            expandformer = TwoSpeedTransformer(
                vocab_size=vocab_size,
                embed_dim=48,          # Smaller start
                hidden_dim=96,         # Smaller start
                context_len=context_len,
                num_blocks=2,
                num_heads=2,
                max_splits=3           # FIXED: Hard limit
            )
        else:
            expandformer = TwoSpeedTransformer(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=128,
                context_len=context_len,
                num_blocks=2,
                num_heads=2
            )

        metrics_expand = runner.train_expandformer(
            expandformer, f"ExpandFormer {version_name}", num_updates, context_len
        )
        all_metrics.append(metrics_expand)

    except ImportError as e:
        print(f"\n⚠️  Could not import ExpandFormer: {e}")
        print("Make sure expandformer_v9.py or expandformer_v10.py is in the current directory")
        print("Skipping ExpandFormer comparison")

    # Generate comparison plots
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    plot_comparison(all_metrics)

    # Print summary
    print("\n" + "="*70)
    print("📈 BENCHMARK SUMMARY")
    print("="*70)

    print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
        "Model", "Test Loss", "Test PPL", "Params", "Time (s)"
    ))
    print("-" * 70)

    for m in all_metrics:
        print("{:<25} {:>12.4f} {:>12.2f} {:>12,} {:>10.1f}".format(
            m.model_name,
            m.final_loss,
            m.final_perplexity,
            m.active_params,
            m.training_time
        ))

    # Determine winner
    print("\n" + "="*70)
    print("🏆 RESULTS")
    print("="*70)

    best_loss = min(all_metrics, key=lambda m: m.final_loss)
    best_efficiency = min(all_metrics, key=lambda m: m.final_loss * m.active_params)

    print(f"\n✓ Best Test Loss: {best_loss.model_name} ({best_loss.final_loss:.4f})")
    print(f"✓ Most Efficient: {best_efficiency.model_name}")

    # Save results
    results_dict = {m.model_name: m.to_dict() for m in all_metrics}
    with open('benchmark_results/results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Saved detailed results to: benchmark_results/results.json")

    print("\n✅ Benchmark complete!")


def main():
    mode = 'quick'

    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            mode = 'full'
        elif sys.argv[1] == '--quick':
            mode = 'quick'
        elif sys.argv[1] == '--help':
            print(__doc__)
            return

    run_benchmark(mode)


if __name__ == "__main__":
    main()