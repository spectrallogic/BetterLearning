"""
ExpandFormer Comprehensive Benchmark Suite
===========================================

Tests ORGANIC INTELLIGENCE capabilities:
1. Domain-specific growth patterns
2. Multi-speed gradient learning
3. Subconscious planning effectiveness
4. Natural abstraction emergence
5. Uneven capacity allocation
6. Adaptive learning behavior

USAGE:
python benchmark_v13.py --quick                    # Fast test, all models
python benchmark_v13.py --full                     # Complete test, all models
python benchmark_v13.py --versions v13 v14         # Test specific versions
python benchmark_v13.py --versions v14 --no-baselines  # Only v14, skip baselines
python benchmark_v13.py --list                     # List available versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict, Counter
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import sys
import argparse
from typing import List, Dict, Tuple

try:
    import tiktoken
except ImportError:
    print("ERROR: pip install tiktoken")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    print("⚠️  sklearn not available - abstraction visualization disabled")
    HAS_SKLEARN = False


# ============================================================================
# BASELINE MODELS (For Comparison)
# ============================================================================

class StaticTransformer(nn.Module):
    """Traditional fixed-size transformer"""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 context_len=256, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._create_pos_encoding(context_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

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

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.output_norm(h)
        logits = self.output(h)

        return logits


# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class GrowthMetrics:
    """Track organic growth behavior"""
    model_name: str
    version: str = ""

    # Growth tracking
    growth_events: List[Dict] = field(default_factory=list)
    domain_sizes: List[int] = field(default_factory=list)
    param_history: List[int] = field(default_factory=list)

    # Performance
    loss_history: List[float] = field(default_factory=list)
    perplexity_history: List[float] = field(default_factory=list)

    # Domain-specific metrics
    tokens_per_domain: Dict[int, List[int]] = field(default_factory=dict)
    domain_creation_times: List[int] = field(default_factory=list)

    # Speed metrics (for multi-speed models)
    speed_performances: Dict[str, List[float]] = field(default_factory=dict)
    speed_weights: List[List[float]] = field(default_factory=list)

    # Abstraction metrics
    cluster_coherence: List[float] = field(default_factory=list)
    concept_separation: List[float] = field(default_factory=list)

    # Final stats
    total_params: int = 0
    active_params: int = 0
    final_loss: float = float('inf')
    final_perplexity: float = float('inf')
    training_time: float = 0.0

    def to_dict(self):
        return {
            'model_name': self.model_name,
            'version': self.version,
            'total_domains': len(self.domain_sizes),
            'growth_events': len(self.growth_events),
            'final_params': self.active_params,
            'param_efficiency': self.final_loss / (self.active_params / 1e6) if self.active_params > 0 else 0,
            'final_loss': self.final_loss,
            'final_perplexity': self.final_perplexity,
            'training_time': self.training_time,
            'growth_pattern': 'organic' if len(self.growth_events) > 0 else 'static',
        }


@dataclass
class TaskMetrics:
    """Performance on specific tasks"""

    # Task 1: Pattern Recognition
    pattern_accuracy: float = 0.0
    pattern_speed: float = 0.0  # Updates to learn

    # Task 2: Novel Generalization
    generalization_score: float = 0.0

    # Task 3: Long-range Dependencies
    long_range_accuracy: float = 0.0

    # Task 4: Rare Token Handling
    rare_token_loss: float = float('inf')

    # Task 5: Domain Adaptation
    adaptation_speed: float = 0.0

    def to_dict(self):
        return {
            'pattern_accuracy': self.pattern_accuracy,
            'pattern_speed': self.pattern_speed,
            'generalization_score': self.generalization_score,
            'long_range_accuracy': self.long_range_accuracy,
            'rare_token_loss': self.rare_token_loss,
            'adaptation_speed': self.adaptation_speed,
            'overall_score': np.mean([
                self.pattern_accuracy,
                self.generalization_score,
                self.long_range_accuracy,
                1.0 / (self.rare_token_loss + 1),
                1.0 / (self.adaptation_speed + 1)
            ])
        }


# ============================================================================
# TASK EVALUATORS
# ============================================================================

class TaskEvaluator:
    """Evaluate models on specific intelligence tasks"""

    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_pattern_recognition(self, model, optimizer, pattern_data):
        """
        Task 1: How fast can model learn simple patterns?
        Pattern: ABABAB... → Should predict A after B, B after A
        """
        model.train()

        updates_to_90_percent = 0
        correct_predictions = 0
        total_predictions = 0

        for update in range(500):  # Max 500 updates
            for sequence in pattern_data:
                tokens = self.tokenizer.encode(sequence)

                for i in range(1, len(tokens)):
                    x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    logits = model(x)

                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]

                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()

                    # Check prediction
                    pred = logits.argmax(dim=-1).item()
                    if pred == y.item():
                        correct_predictions += 1
                    total_predictions += 1

            # Check if reached 90% accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            if accuracy >= 0.90 and updates_to_90_percent == 0:
                updates_to_90_percent = update + 1
                break

        model.eval()
        return accuracy, updates_to_90_percent if updates_to_90_percent > 0 else 500

    def evaluate_generalization(self, model, train_examples, test_examples):
        """
        Task 2: Train on some examples, test on novel variations
        Tests abstraction capability
        """
        model.eval()

        total_loss = 0
        num_predictions = 0

        with torch.no_grad():
            for sequence in test_examples:
                tokens = self.tokenizer.encode(sequence)

                for i in range(1, len(tokens)):
                    x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    logits = model(x)
                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]

                    loss = F.cross_entropy(logits, y)
                    total_loss += loss.item()
                    num_predictions += 1

        avg_loss = total_loss / num_predictions if num_predictions > 0 else float('inf')

        # Convert to score (lower loss = better)
        score = math.exp(-avg_loss / 5.0)  # Normalize

        return score

    def evaluate_long_range(self, model, sequences_with_deps):
        """
        Task 3: Long-range dependencies
        Tests if model can connect info across many tokens
        """
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for sequence, dependency_position in sequences_with_deps:
                tokens = self.tokenizer.encode(sequence)

                if dependency_position >= len(tokens):
                    continue

                # Test if model predicts correctly based on early context
                x = torch.tensor([tokens[:dependency_position]], dtype=torch.long, device=self.device)
                y = torch.tensor([tokens[dependency_position]], dtype=torch.long, device=self.device)

                logits = model(x)
                if len(logits.shape) > 2:
                    logits = logits[:, -1, :]

                pred = logits.argmax(dim=-1).item()
                if pred == y.item():
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def evaluate_rare_tokens(self, model, rare_token_sequences):
        """
        Task 4: How well does model handle rare/difficult tokens?
        """
        model.eval()

        total_loss = 0
        num_predictions = 0

        with torch.no_grad():
            for sequence in rare_token_sequences:
                tokens = self.tokenizer.encode(sequence)

                for i in range(1, len(tokens)):
                    x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    logits = model(x)
                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]

                    loss = F.cross_entropy(logits, y)
                    total_loss += loss.item()
                    num_predictions += 1

        avg_loss = total_loss / num_predictions if num_predictions > 0 else float('inf')
        return avg_loss

    def evaluate_adaptation(self, model, optimizer, new_domain_data):
        """
        Task 5: How fast does model adapt to new domain/style?
        """
        model.train()

        initial_losses = []
        final_losses = []

        # Initial performance
        for sequence in new_domain_data[:5]:
            tokens = self.tokenizer.encode(sequence)
            for i in range(1, len(tokens)):
                x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    logits = model(x)
                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]
                    loss = F.cross_entropy(logits, y)
                    initial_losses.append(loss.item())

        # Train for 50 updates
        for _ in range(50):
            for sequence in new_domain_data:
                tokens = self.tokenizer.encode(sequence)
                for i in range(1, len(tokens)):
                    x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                    y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    logits = model(x)
                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()

        # Final performance
        model.eval()
        for sequence in new_domain_data[:5]:
            tokens = self.tokenizer.encode(sequence)
            for i in range(1, len(tokens)):
                x = torch.tensor([tokens[:i]], dtype=torch.long, device=self.device)
                y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    logits = model(x)
                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]
                    loss = F.cross_entropy(logits, y)
                    final_losses.append(loss.item())

        model.train()

        # Adaptation speed = how much it improved
        initial_avg = np.mean(initial_losses)
        final_avg = np.mean(final_losses)
        improvement = initial_avg - final_avg

        return max(0, improvement)


# ============================================================================
# GROWTH ANALYZER
# ============================================================================

class GrowthAnalyzer:
    """Analyze organic growth patterns"""

    def __init__(self):
        self.snapshots = []

    def take_snapshot(self, model, update_num):
        """Capture model state for later analysis"""
        snapshot = {
            'update': update_num,
            'total_params': sum(p.numel() for p in model.parameters()),
            'active_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Check for various growth attributes
        # v13-style
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'domain_expansions'):
            snapshot['num_domains'] = len(model.embeddings.domain_expansions)
            snapshot['domain_sizes'] = [
                sum(p.numel() for p in domain.parameters())
                for domain in model.embeddings.domain_expansions
            ]
            domain_token_counts = [len(tokens) for tokens in model.embeddings.domain_token_sets]
            snapshot['tokens_per_domain'] = domain_token_counts

        # v14-style
        if hasattr(model, 'embed') and hasattr(model.embed, 'domains'):
            snapshot['num_domains'] = len(model.embed.domains)
            snapshot['domain_sizes'] = [
                sum(p.numel() for p in domain.parameters())
                for domain in model.embed.domains
            ]
            if hasattr(model.embed, 'domain_token_sets'):
                domain_token_counts = [len(tokens) for tokens in model.embed.domain_token_sets]
                snapshot['tokens_per_domain'] = domain_token_counts

        # Multi-speed weights
        if hasattr(model, 'multi_speed') and hasattr(model.multi_speed, 'speed_weights'):
            snapshot['speed_weights'] = model.multi_speed.speed_weights.detach().cpu().numpy().tolist()

        # Transformer blocks growth
        if hasattr(model, 'blocks'):
            snapshot['num_blocks'] = len(model.blocks)

        self.snapshots.append(snapshot)

    def analyze_growth_pattern(self):
        """Determine if growth is organic (uneven) or uniform"""
        if len(self.snapshots) < 2:
            return "insufficient_data"

        # Check domain sizes for unevenness
        final_snapshot = self.snapshots[-1]

        if 'domain_sizes' not in final_snapshot or len(final_snapshot['domain_sizes']) == 0:
            return "no_growth"

        domain_sizes = final_snapshot['domain_sizes']

        # Calculate coefficient of variation (std/mean)
        if len(domain_sizes) > 1:
            cv = np.std(domain_sizes) / (np.mean(domain_sizes) + 1e-8)

            if cv > 0.3:
                return "organic_uneven"  # Good! Uneven growth
            elif cv > 0.1:
                return "somewhat_uneven"
            else:
                return "uniform"  # Not ideal

        return "single_domain"

    def get_growth_summary(self):
        """Generate growth summary statistics"""
        if len(self.snapshots) == 0:
            return {}

        initial = self.snapshots[0]
        final = self.snapshots[-1]

        summary = {
            'total_snapshots': len(self.snapshots),
            'initial_params': initial['total_params'],
            'final_params': final['total_params'],
            'param_growth': final['total_params'] - initial['total_params'],
            'growth_rate': (final['total_params'] - initial['total_params']) / initial['total_params'] if initial['total_params'] > 0 else 0,
        }

        if 'num_domains' in final:
            summary['domains_created'] = final['num_domains']
            summary['avg_domain_size'] = np.mean(final['domain_sizes']) if final['domain_sizes'] else 0
            summary['domain_size_std'] = np.std(final['domain_sizes']) if len(final['domain_sizes']) > 1 else 0
            summary['growth_pattern'] = self.analyze_growth_pattern()

        if 'num_blocks' in final:
            summary['blocks_added'] = final['num_blocks'] - initial.get('num_blocks', 0)

        return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

class BenchmarkVisualizer:
    """Create comprehensive visualizations"""

    def __init__(self, save_dir='benchmark_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_growth_comparison(self, growth_histories):
        """Compare growth patterns across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Parameter growth over time
        ax = axes[0, 0]
        for model_name, history in growth_histories.items():
            if history['param_history']:
                ax.plot(history['param_history'], label=model_name, linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Total Parameters')
        ax.set_title('Parameter Growth Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss over time
        ax = axes[0, 1]
        for model_name, history in growth_histories.items():
            if history['loss_history']:
                ax.plot(history['loss_history'], label=model_name, linewidth=2)
        ax.set_xlabel('Update (x100)')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Domain creation events
        ax = axes[1, 0]
        for model_name, history in growth_histories.items():
            if history['growth_events']:
                events = history['growth_events']
                updates = [e['update'] for e in events]
                domains = list(range(1, len(updates) + 1))
                ax.scatter(updates, domains, label=model_name, s=100, alpha=0.7)
        ax.set_xlabel('Update')
        ax.set_ylabel('Cumulative Domains Created')
        ax.set_title('Domain Creation Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Parameter efficiency
        ax = axes[1, 1]
        models = []
        final_losses = []
        final_params = []
        for model_name, history in growth_histories.items():
            if history['loss_history'] and history['param_history']:
                models.append(model_name)
                final_losses.append(history['loss_history'][-1])
                final_params.append(history['param_history'][-1] / 1e6)

        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        for i, (name, loss, params, color) in enumerate(zip(models, final_losses, final_params, colors)):
            ax.scatter(params, loss, s=200, c=[color], label=name, alpha=0.7, edgecolors='black')
        ax.set_xlabel('Parameters (Millions)')
        ax.set_ylabel('Final Loss')
        ax.set_title('Parameter Efficiency (Lower-Left is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'growth_comparison.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {self.save_dir / 'growth_comparison.png'}")
        plt.close()

    def plot_domain_analysis(self, metrics):
        """Detailed analysis of domain growth"""
        if not metrics.domain_sizes:
            print("⚠️  No domain data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Domain sizes
        ax = axes[0, 0]
        domain_ids = list(range(len(metrics.domain_sizes)))
        ax.bar(domain_ids, metrics.domain_sizes, color='skyblue', edgecolor='black')
        ax.set_xlabel('Domain ID')
        ax.set_ylabel('Size (parameters)')
        ax.set_title('Domain Sizes (Unevenness = Organic Growth)')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Tokens per domain
        ax = axes[0, 1]
        if metrics.tokens_per_domain:
            domain_ids = list(metrics.tokens_per_domain.keys())
            token_counts = [len(tokens) for tokens in metrics.tokens_per_domain.values()]
            ax.bar(domain_ids, token_counts, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Domain ID')
            ax.set_ylabel('Number of Tokens')
            ax.set_title('Tokens Assigned to Each Domain')
            ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Domain creation timeline
        ax = axes[1, 0]
        if metrics.domain_creation_times:
            ax.plot(metrics.domain_creation_times, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Domain ID')
            ax.set_ylabel('Creation Time (update)')
            ax.set_title('When Were Domains Created?')
            ax.grid(True, alpha=0.3)

        # Plot 4: Growth events detail
        ax = axes[1, 1]
        if metrics.growth_events:
            updates = [e['update'] for e in metrics.growth_events]
            new_params = [e.get('new_params', 0) for e in metrics.growth_events]
            ax.scatter(updates, new_params, s=100, alpha=0.7, color='green', edgecolors='black')
            ax.set_xlabel('Update')
            ax.set_ylabel('New Parameters Added')
            ax.set_title('Parameter Addition at Each Growth Event')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'domain_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {self.save_dir / 'domain_analysis.png'}")
        plt.close()

    def plot_speed_analysis(self, metrics):
        """Analyze multi-speed learning performance"""
        if not metrics.speed_weights:
            print("⚠️  No speed data to visualize")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Speed weight evolution
        ax = axes[0]
        speed_weights = np.array(metrics.speed_weights)
        num_speeds = speed_weights.shape[1]

        for i in range(num_speeds):
            ax.plot(speed_weights[:, i], label=f'Speed {i}', linewidth=2)
        ax.set_xlabel('Update (x100)')
        ax.set_ylabel('Weight')
        ax.set_title('Multi-Speed Weight Adaptation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Speed performances
        ax = axes[1]
        if metrics.speed_performances:
            for speed_name, losses in metrics.speed_performances.items():
                ax.plot(losses, label=speed_name, linewidth=2, alpha=0.7)
            ax.set_xlabel('Update (x100)')
            ax.set_ylabel('Loss')
            ax.set_title('Per-Speed Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'speed_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {self.save_dir / 'speed_analysis.png'}")
        plt.close()

    def plot_task_comparison(self, task_results):
        """Compare models on specific tasks"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        models = list(task_results.keys())

        # Task 1: Pattern Recognition
        ax = axes[0, 0]
        accuracies = [task_results[m].pattern_accuracy for m in models]
        ax.bar(models, accuracies, color='skyblue', edgecolor='black')
        ax.set_ylabel('Accuracy')
        ax.set_title('Pattern Recognition')
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Task 2: Pattern Learning Speed
        ax = axes[0, 1]
        speeds = [task_results[m].pattern_speed for m in models]
        ax.bar(models, speeds, color='lightcoral', edgecolor='black')
        ax.set_ylabel('Updates to 90%')
        ax.set_title('Learning Speed (Lower is Better)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Task 3: Generalization
        ax = axes[0, 2]
        gen_scores = [task_results[m].generalization_score for m in models]
        ax.bar(models, gen_scores, color='lightgreen', edgecolor='black')
        ax.set_ylabel('Score')
        ax.set_title('Generalization Ability')
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Task 4: Long-range Dependencies
        ax = axes[1, 0]
        long_range = [task_results[m].long_range_accuracy for m in models]
        ax.bar(models, long_range, color='plum', edgecolor='black')
        ax.set_ylabel('Accuracy')
        ax.set_title('Long-Range Dependencies')
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Task 5: Rare Token Handling
        ax = axes[1, 1]
        rare_losses = [task_results[m].rare_token_loss for m in models]
        ax.bar(models, rare_losses, color='khaki', edgecolor='black')
        ax.set_ylabel('Loss')
        ax.set_title('Rare Token Loss (Lower is Better)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Task 6: Adaptation Speed
        ax = axes[1, 2]
        adapt_speeds = [task_results[m].adaptation_speed for m in models]
        ax.bar(models, adapt_speeds, color='lightblue', edgecolor='black')
        ax.set_ylabel('Improvement')
        ax.set_title('Domain Adaptation (Higher is Better)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'task_comparison.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {self.save_dir / 'task_comparison.png'}")
        plt.close()

    def generate_summary_report(self, all_metrics, task_results, growth_summaries):
        """Generate comprehensive text report"""
        report_path = self.save_dir / 'benchmark_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("EXPANDFORMER BENCHMARK REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Overall Performance
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            for model_name, metrics in all_metrics.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Final Loss: {metrics.final_loss:.4f}\n")
                f.write(f"  Final Perplexity: {metrics.final_perplexity:.2f}\n")
                f.write(f"  Total Parameters: {metrics.active_params:,}\n")
                f.write(f"  Training Time: {metrics.training_time:.1f}s\n")

                if metrics.version:
                    f.write(f"  Domains Created: {len(metrics.domain_sizes)}\n")
                    f.write(f"  Growth Events: {len(metrics.growth_events)}\n")

            # Task Performance
            f.write("\n\nTASK-SPECIFIC PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            for model_name, task_metric in task_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Pattern Recognition: {task_metric.pattern_accuracy:.3f}\n")
                f.write(f"  Learning Speed: {task_metric.pattern_speed} updates\n")
                f.write(f"  Generalization: {task_metric.generalization_score:.3f}\n")
                f.write(f"  Long-Range Deps: {task_metric.long_range_accuracy:.3f}\n")
                f.write(f"  Rare Token Loss: {task_metric.rare_token_loss:.4f}\n")
                f.write(f"  Adaptation Speed: {task_metric.adaptation_speed:.4f}\n")
                f.write(f"  Overall Score: {task_metric.to_dict()['overall_score']:.3f}\n")

            # Growth Analysis
            f.write("\n\nGROWTH ANALYSIS\n")
            f.write("-" * 70 + "\n")
            for model_name, summary in growth_summaries.items():
                if summary:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Initial Params: {summary.get('initial_params', 0):,}\n")
                    f.write(f"  Final Params: {summary.get('final_params', 0):,}\n")
                    f.write(f"  Growth Rate: {summary.get('growth_rate', 0):.2%}\n")
                    if 'domains_created' in summary:
                        f.write(f"  Domains Created: {summary['domains_created']}\n")
                        f.write(f"  Growth Pattern: {summary.get('growth_pattern', 'N/A')}\n")
                    if 'blocks_added' in summary:
                        f.write(f"  Blocks Added: {summary['blocks_added']}\n")

            # Winner Determination
            f.write("\n\nRESULTS\n")
            f.write("=" * 70 + "\n")

            # Best overall loss
            best_loss = min(all_metrics.items(), key=lambda x: x[1].final_loss)
            f.write(f"\n🏆 Best Loss: {best_loss[0]} ({best_loss[1].final_loss:.4f})\n")

            # Best efficiency (loss per million params)
            best_efficiency = min(all_metrics.items(),
                                  key=lambda x: x[1].final_loss * x[1].active_params)
            f.write(f"🏆 Most Efficient: {best_efficiency[0]}\n")

            # Best on tasks
            if task_results:
                best_tasks = max(task_results.items(),
                                 key=lambda x: x[1].to_dict()['overall_score'])
                f.write(f"🏆 Best Task Performance: {best_tasks[0]}\n")

            # Organic growth achievement
            for model_name, summary in growth_summaries.items():
                if summary.get('growth_pattern') == 'organic_uneven':
                    f.write(f"🌱 Organic Growth Achieved: {model_name}\n")

        print(f"📄 Saved: {report_path}")


# ============================================================================
# VERSION LOADER
# ============================================================================

class VersionLoader:
    """Load different ExpandFormer versions dynamically"""

    AVAILABLE_VERSIONS = {
        'v13': {
            'module': 'expandformer_v13',
            'class': 'OrganicGrowthTransformer',
            'description': 'Organic Growth with Subconscious Planning',
            'config': {
                'base_dim': 64,
                'hidden_dim': 128,
                'num_blocks': 2,
                'num_heads': 2,
                'max_domains': 20
            }
        },
        'v14': {
            'module': 'expandformer_v14',
            'class': 'MicroscopicGrowthTransformer',
            'description': 'Microscopic Start (Tiny → Grows)',
            'config': {
                'base_dim': 8,
                'hidden_dim': 16,
                'num_heads': 1,
                'max_domains': 30,
                'max_blocks': 10
            }
        },
        'v12': {
            'module': 'expandformer_v12',
            'class': 'ExpandFormer',
            'description': 'Forced Abstraction (16→32 dims)',
            'config': {
                'base_dim': 16,
                'hidden_dim': 32,
                'num_blocks': 2,
                'num_heads': 1
            }
        }
    }

    @classmethod
    def list_versions(cls):
        """Print available versions"""
        print("\n" + "=" * 70)
        print("AVAILABLE EXPANDFORMER VERSIONS")
        print("=" * 70)
        for version, info in cls.AVAILABLE_VERSIONS.items():
            print(f"\n{version}: {info['description']}")
            print(f"  Module: {info['module']}.py")
            print(f"  Class: {info['class']}")
            print(f"  Config: {info['config']}")
        print("\n" + "=" * 70 + "\n")

    @classmethod
    def load_version(cls, version, vocab_size, context_len):
        """
        Load a specific version
        Returns: (model, version_name) or (None, None) if failed
        """
        if version not in cls.AVAILABLE_VERSIONS:
            print(f"⚠️  Unknown version: {version}")
            print(f"Available versions: {list(cls.AVAILABLE_VERSIONS.keys())}")
            return None, None

        info = cls.AVAILABLE_VERSIONS[version]

        try:
            # Try to import the module
            sys.path.insert(0, str(Path.cwd()))
            module = __import__(info['module'])
            model_class = getattr(module, info['class'])

            # Create model with config
            config = info['config'].copy()
            config['vocab_size'] = vocab_size
            config['context_len'] = context_len

            model = model_class(**config)

            print(f"✓ Loaded {version}: {info['description']}")

            return model, version

        except ImportError as e:
            print(f"\n⚠️  Could not import {version}: {e}")
            print(f"Make sure {info['module']}.py exists in current directory")
            return None, None
        except Exception as e:
            print(f"\n⚠️  Error creating {version}: {e}")
            return None, None


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

class ComprehensiveBenchmark:
    """Run complete benchmark suite"""

    def __init__(self, training_texts, test_texts, device='cuda'):
        self.training_texts = training_texts
        self.test_texts = test_texts
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.task_evaluator = TaskEvaluator(self.tokenizer, device)
        self.visualizer = BenchmarkVisualizer()

        # Prepare task-specific data
        self._prepare_task_data()

    def _prepare_task_data(self):
        """Prepare specialized datasets for each task"""

        # Task 1: Pattern recognition
        self.pattern_data = [
            "A B A B A B A B",
            "X Y X Y X Y X Y",
            "1 2 1 2 1 2 1 2",
        ]

        # Task 2: Generalization (train on some, test on variations)
        self.generalization_train = self.training_texts[:10]
        self.generalization_test = self.training_texts[10:15]

        # Task 3: Long-range dependencies
        self.long_range_data = [
            ("The cat sat on the mat and later the cat", 9),
            ("In the beginning there was light and darkness", 8),
        ]

        # Task 4: Rare tokens (technical jargon, uncommon words)
        self.rare_token_data = [
            "The mitochondria is the powerhouse of the cell",
            "Quantum entanglement exhibits non-local correlations",
            "Photosynthesis converts luminous energy into chemical energy",
        ]

        # Task 5: New domain adaptation (different style)
        self.new_domain_data = [
            "import torch\nimport numpy as np\nfrom pathlib import Path",
            "def train_model(x, y):\n    optimizer.zero_grad()\n    loss.backward()",
            "class NeuralNetwork(nn.Module):\n    def __init__(self):",
        ]

    def train_and_evaluate(self, model, model_name, version="", num_updates=1000, context_len=128):
        """Train model and collect comprehensive metrics"""
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'=' * 70}\n")

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

        # Initialize metrics
        metrics = GrowthMetrics(model_name=model_name, version=version)
        metrics.total_params = sum(p.numel() for p in model.parameters())

        # Initialize growth analyzer
        growth_analyzer = GrowthAnalyzer()

        start_time = time.time()
        update_count = 0
        recent_losses = deque(maxlen=100)

        print(f"Initial parameters: {metrics.total_params:,}\n")

        # Training loop with comprehensive tracking
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
                    y_token = tokens[i]
                    y = torch.tensor([y_token], dtype=torch.long, device=self.device)

                    # Forward pass
                    optimizer.zero_grad()

                    # Handle different return types
                    output = model(x)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                    if len(logits.shape) > 2:
                        logits = logits[:, -1, :]

                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Track metrics
                    recent_losses.append(loss.item())
                    update_count += 1

                    # Update tracking for growth models
                    if hasattr(model, 'update_tracking'):
                        model.update_tracking(loss.item(), y_token)
                    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'update_difficulty'):
                        model.embeddings.update_difficulty(y_token, loss.item())
                    elif hasattr(model, 'embed') and hasattr(model.embed, 'update_difficulty'):
                        model.embed.update_difficulty(y_token, loss.item())

                    # Check for growth
                    if hasattr(model, 'check_and_grow') and update_count % 50 == 0:
                        if model.check_and_grow():
                            new_params = sum(p.numel() for p in model.parameters())
                            metrics.growth_events.append({
                                'update': update_count,
                                'new_params': new_params - metrics.param_history[-1] if metrics.param_history else 0
                            })
                            metrics.domain_creation_times.append(update_count)

                    # Periodic tracking
                    if update_count % 100 == 0:
                        avg_loss = np.mean(list(recent_losses))
                        perplexity = math.exp(min(avg_loss, 20))

                        metrics.loss_history.append(avg_loss)
                        metrics.perplexity_history.append(perplexity)
                        metrics.param_history.append(sum(p.numel() for p in model.parameters()))

                        # Take growth snapshot
                        growth_analyzer.take_snapshot(model, update_count)

                        # Track domain metrics (v13-style)
                        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'domain_expansions'):
                            metrics.domain_sizes = [
                                sum(p.numel() for p in domain.parameters())
                                for domain in model.embeddings.domain_expansions
                            ]
                            if hasattr(model.embeddings, 'domain_token_sets'):
                                metrics.tokens_per_domain = {
                                    i: list(tokens) for i, tokens in enumerate(model.embeddings.domain_token_sets)
                                }

                        # Track domain metrics (v14-style)
                        if hasattr(model, 'embed') and hasattr(model.embed, 'domains'):
                            metrics.domain_sizes = [
                                sum(p.numel() for p in domain.parameters())
                                for domain in model.embed.domains
                            ]
                            if hasattr(model.embed, 'domain_token_sets'):
                                metrics.tokens_per_domain = {
                                    i: list(tokens) for i, tokens in enumerate(model.embed.domain_token_sets)
                                }

                        # Track multi-speed weights
                        if hasattr(model, 'multi_speed') and hasattr(model.multi_speed, 'speed_weights'):
                            weights = model.multi_speed.speed_weights.detach().cpu().numpy()
                            metrics.speed_weights.append(weights.tolist())

                        print(f"Update {update_count:5d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                              f"Params: {metrics.param_history[-1]:,}")

                    if update_count >= num_updates:
                        break

                if update_count >= num_updates:
                    break

        metrics.training_time = time.time() - start_time
        metrics.active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics.final_loss = metrics.loss_history[-1] if metrics.loss_history else float('inf')
        metrics.final_perplexity = metrics.perplexity_history[-1] if metrics.perplexity_history else float('inf')

        print(f"\n✓ Training complete!")
        print(f"  Final loss: {metrics.final_loss:.4f}")
        print(f"  Final params: {metrics.active_params:,}")
        print(f"  Training time: {metrics.training_time:.1f}s\n")

        # Growth analysis summary
        growth_summary = growth_analyzer.get_growth_summary()

        return metrics, growth_summary

    def evaluate_tasks(self, model, model_name):
        """Evaluate model on all specialized tasks"""
        print(f"\n{'=' * 70}")
        print(f"TASK EVALUATION: {model_name}")
        print(f"{'=' * 70}\n")

        task_metrics = TaskMetrics()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

        # Task 1: Pattern Recognition
        print("Task 1: Pattern Recognition...")
        accuracy, speed = self.task_evaluator.evaluate_pattern_recognition(
            model, optimizer, self.pattern_data
        )
        task_metrics.pattern_accuracy = accuracy
        task_metrics.pattern_speed = speed
        print(f"  Accuracy: {accuracy:.3f} | Speed: {speed} updates\n")

        # Task 2: Generalization
        print("Task 2: Generalization...")
        gen_score = self.task_evaluator.evaluate_generalization(
            model, self.generalization_train, self.generalization_test
        )
        task_metrics.generalization_score = gen_score
        print(f"  Score: {gen_score:.3f}\n")

        # Task 3: Long-range Dependencies
        print("Task 3: Long-Range Dependencies...")
        long_range_acc = self.task_evaluator.evaluate_long_range(
            model, self.long_range_data
        )
        task_metrics.long_range_accuracy = long_range_acc
        print(f"  Accuracy: {long_range_acc:.3f}\n")

        # Task 4: Rare Tokens
        print("Task 4: Rare Token Handling...")
        rare_loss = self.task_evaluator.evaluate_rare_tokens(
            model, self.rare_token_data
        )
        task_metrics.rare_token_loss = rare_loss
        print(f"  Loss: {rare_loss:.4f}\n")

        # Task 5: Domain Adaptation
        print("Task 5: Domain Adaptation...")
        adapt_speed = self.task_evaluator.evaluate_adaptation(
            model, optimizer, self.new_domain_data
        )
        task_metrics.adaptation_speed = adapt_speed
        print(f"  Improvement: {adapt_speed:.4f}\n")

        print(f"Overall Task Score: {task_metrics.to_dict()['overall_score']:.3f}\n")

        return task_metrics

    def run_full_benchmark(self, mode='quick', versions=None, include_baselines=True):
        """Run complete benchmark suite"""
        print("=" * 70)
        print("🔬 EXPANDFORMER COMPREHENSIVE BENCHMARK")
        print("=" * 70)
        print(f"\nMode: {mode}")
        print(f"Device: {self.device}")
        if versions:
            print(f"Testing versions: {', '.join(versions)}")
        if not include_baselines:
            print("Skipping baseline models")
        print()

        # Set parameters based on mode
        if mode == 'quick':
            num_updates = 500
            context_len = 64
        else:
            num_updates = 2000
            context_len = 128

        vocab_size = self.tokenizer.n_vocab

        all_metrics = {}
        all_growth_summaries = {}
        task_results = {}
        growth_histories = {}

        # Run baselines if requested
        if include_baselines:
            # 1. Baseline: Static Small
            print("\n" + "=" * 70)
            print("MODEL: Static Transformer (Small)")
            print("=" * 70)
            baseline_small = StaticTransformer(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=128,
                num_layers=2,
                num_heads=2
            )
            metrics_small, growth_small = self.train_and_evaluate(
                baseline_small, "Static Small", "", num_updates, context_len
            )
            all_metrics["Static Small"] = metrics_small
            all_growth_summaries["Static Small"] = growth_small
            growth_histories["Static Small"] = {
                'param_history': metrics_small.param_history,
                'loss_history': metrics_small.loss_history,
                'growth_events': metrics_small.growth_events
            }

            # Task evaluation
            task_small = self.evaluate_tasks(baseline_small, "Static Small")
            task_results["Static Small"] = task_small

            # 2. Baseline: Static Large
            print("\n" + "=" * 70)
            print("MODEL: Static Transformer (Large)")
            print("=" * 70)
            baseline_large = StaticTransformer(
                vocab_size=vocab_size,
                embed_dim=96,
                hidden_dim=192,
                num_layers=3,
                num_heads=4
            )
            metrics_large, growth_large = self.train_and_evaluate(
                baseline_large, "Static Large", "", num_updates, context_len
            )
            all_metrics["Static Large"] = metrics_large
            all_growth_summaries["Static Large"] = growth_large
            growth_histories["Static Large"] = {
                'param_history': metrics_large.param_history,
                'loss_history': metrics_large.loss_history,
                'growth_events': metrics_large.growth_events
            }

            # Task evaluation
            task_large = self.evaluate_tasks(baseline_large, "Static Large")
            task_results["Static Large"] = task_large

        # Load and test requested versions
        if not versions:
            versions = ['v13', 'v14']  # Default versions

        for version in versions:
            print("\n" + "=" * 70)
            print(f"MODEL: ExpandFormer {version}")
            print("=" * 70)

            model, ver_name = VersionLoader.load_version(version, vocab_size, context_len)

            if model is None:
                print(f"Skipping {version}\n")
                continue

            model_name = f"ExpandFormer {version}"

            metrics, growth_summary = self.train_and_evaluate(
                model, model_name, version, num_updates, context_len
            )
            all_metrics[model_name] = metrics
            all_growth_summaries[model_name] = growth_summary
            growth_histories[model_name] = {
                'param_history': metrics.param_history,
                'loss_history': metrics.loss_history,
                'growth_events': metrics.growth_events
            }

            # Task evaluation
            task_metrics = self.evaluate_tasks(model, model_name)
            task_results[model_name] = task_metrics

        # Generate all visualizations
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70 + "\n")

        self.visualizer.plot_growth_comparison(growth_histories)

        # Plot domain analysis for any model that has domains
        for model_name, metrics in all_metrics.items():
            if metrics.domain_sizes:
                self.visualizer.plot_domain_analysis(metrics)
                self.visualizer.plot_speed_analysis(metrics)
                break  # Only plot once

        self.visualizer.plot_task_comparison(task_results)
        self.visualizer.generate_summary_report(all_metrics, task_results, all_growth_summaries)

        # Print summary
        print("\n" + "=" * 70)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 70 + "\n")

        for model_name, metrics in all_metrics.items():
            print(f"{model_name}:")
            print(f"  Loss: {metrics.final_loss:.4f} | PPL: {metrics.final_perplexity:.2f}")
            print(f"  Params: {metrics.active_params:,} | Time: {metrics.training_time:.1f}s")
            if model_name in task_results:
                task_score = task_results[model_name].to_dict()['overall_score']
                print(f"  Task Score: {task_score:.3f}")
            print()

        print("=" * 70)
        print("🏆 WINNERS")
        print("=" * 70 + "\n")

        best_loss = min(all_metrics.items(), key=lambda x: x[1].final_loss)
        print(f"Best Loss: {best_loss[0]} ({best_loss[1].final_loss:.4f})")

        best_efficiency = min(all_metrics.items(),
                              key=lambda x: x[1].final_loss * x[1].active_params)
        print(f"Most Efficient: {best_efficiency[0]}")

        if task_results:
            best_tasks = max(task_results.items(),
                             key=lambda x: x[1].to_dict()['overall_score'])
            print(f"Best Tasks: {best_tasks[0]} ({best_tasks[1].to_dict()['overall_score']:.3f})")

        # Check for organic growth
        for model_name, summary in all_growth_summaries.items():
            if summary.get('growth_pattern') == 'organic_uneven':
                print(f"\n🌱 Organic Growth Achieved: {model_name}")
                print(f"   Pattern: Uneven domain sizes (natural specialization)")

        print("\n✅ Benchmark complete!")
        print(f"📁 Results saved to: {self.visualizer.save_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ExpandFormer Comprehensive Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick test (500 updates, ~5 min)')
    parser.add_argument('--full', action='store_true',
                       help='Full test (2000 updates, ~20 min)')
    parser.add_argument('--versions', nargs='+',
                       help='Specific versions to test (e.g., v13 v14)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline static models')
    parser.add_argument('--list', action='store_true',
                       help='List available versions and exit')

    args = parser.parse_args()

    # List versions if requested
    if args.list:
        VersionLoader.list_versions()
        return

    # Determine mode
    if args.full:
        mode = 'full'
    else:
        mode = 'quick'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                    chunk = '\n'.join(lines[i:i + 6])
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
                    ] * 20

    # Split train/test
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    test_texts = all_texts[split_idx:]

    print(f"✓ Train: {len(train_texts)} chunks")
    print(f"✓ Test: {len(test_texts)} chunks\n")

    # Run benchmark
    benchmark = ComprehensiveBenchmark(train_texts, test_texts, device=device)
    benchmark.run_full_benchmark(
        mode=mode,
        versions=args.versions,
        include_baselines=not args.no_baselines
    )


if __name__ == "__main__":
    main()