"""
ExpandFormer v15: COMPLETE Attention-Organic ASI Architecture
==============================================================

THE FULL VISION:
- ✅ Multi-Speed Gradient Learning (16, 32, 64, 128 dims)
- ✅ Subconscious Planning (4-layer predictive system)
- ✅ Attention-Based Hierarchy Discovery (NEW)
- ✅ Organic Domain Growth (NEW - actually works)
- ✅ Domain-Specific Embeddings
- ✅ Natural Abstraction

PHILOSOPHY:
"Start tiny, grow organically, learn at multiple speeds,
plan subconsciously, discover hierarchy through attention"

USAGE:
python expandformer_v15.py              # Train
python expandformer_v15.py --benchmark  # Benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque, defaultdict
from pathlib import Path
import sys
import random

try:
    import tiktoken
except ImportError:
    print("ERROR: pip install tiktoken")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    print("⚠️  sklearn not available - clustering simplified")
    HAS_SKLEARN = False


# ============================================================================
# ATTENTION-BASED HIERARCHY DISCOVERY
# ============================================================================

class AttentionAnalyzer:
    """
    Analyzes attention patterns to discover natural hierarchy
    NO manual classification - pure attention flow
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # Store attention patterns per token
        self.token_attention = defaultdict(lambda: {
            'incoming': [],
            'outgoing': [],
            'self_attn': [],
        })

        # Recent attention snapshots
        self.attention_snapshots = deque(maxlen=200)

    def capture_attention(self, attention_weights, token_ids):
        """
        Capture attention patterns from forward pass

        attention_weights: [batch, num_heads, seq_len, seq_len]
        token_ids: [batch, seq_len]
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Average over batch and heads
        attn = attention_weights.mean(dim=0).mean(dim=0)  # [seq_len, seq_len]

        # Store snapshot
        self.attention_snapshots.append({
            'weights': attn.detach().cpu(),
            'tokens': token_ids[0].detach().cpu()
        })

        # Update per-token statistics
        tokens = token_ids[0].cpu().numpy()

        for i, token_id in enumerate(tokens):
            if token_id >= self.vocab_size or token_id < 0:
                continue

            # Incoming: how much others attend TO this token
            incoming = attn[:, i].mean().item()
            self.token_attention[token_id]['incoming'].append(incoming)

            # Outgoing: how much this token attends TO others
            outgoing = attn[i, :].mean().item()
            self.token_attention[token_id]['outgoing'].append(outgoing)

            # Self-attention
            self_attn = attn[i, i].item()
            self.token_attention[token_id]['self_attn'].append(self_attn)

    def compute_attention_properties(self, min_samples=10):
        """
        Compute attention-based properties for each token
        """
        properties = {}

        for token_id, patterns in self.token_attention.items():
            if len(patterns['incoming']) < min_samples:
                continue

            incoming = np.array(patterns['incoming'])
            outgoing = np.array(patterns['outgoing'])
            self_attn = np.array(patterns['self_attn'])

            properties[token_id] = {
                'incoming_mean': incoming.mean(),
                'outgoing_mean': outgoing.mean(),
                'self_mean': self_attn.mean(),
                'incoming_std': incoming.std(),
                'outgoing_std': outgoing.std(),
                'attention_ratio': incoming.mean() / (outgoing.mean() + 1e-6),
                'stability': 1.0 / (incoming.std() + 1e-6),
                'n_samples': len(incoming)
            }

        return properties

    def find_attention_clusters(self, properties, loss_dict=None):
        """
        Cluster tokens by attention patterns
        """
        if not properties or len(properties) < 3:
            return []

        token_ids = []
        features = []

        for token_id, props in properties.items():
            if props['n_samples'] >= 10:
                token_ids.append(token_id)

                feature = [
                    props['incoming_mean'],
                    props['outgoing_mean'],
                    props['self_mean'],
                    props['attention_ratio'],
                    props['stability'],
                    props['incoming_std'],
                ]

                if loss_dict and token_id in loss_dict:
                    feature.append(loss_dict[token_id])

                features.append(feature)

        if len(features) < 3:
            return []

        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

        # Cluster
        if HAS_SKLEARN:
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(features)
            labels = clustering.labels_
        else:
            ratios = features[:, 3]
            labels = np.digitize(ratios, bins=[0.5, 1.0, 2.0]) - 1

        clusters = defaultdict(list)
        for token_id, label in zip(token_ids, labels):
            if label != -1:
                clusters[label].append(token_id)

        return [tokens for tokens in clusters.values() if len(tokens) >= 3]

    def get_cluster_properties(self, cluster_tokens, properties):
        """
        Aggregate properties for a cluster
        """
        if not cluster_tokens:
            return None

        cluster_props = defaultdict(list)

        for token_id in cluster_tokens:
            if token_id in properties:
                props = properties[token_id]
                for key, value in props.items():
                    if isinstance(value, (int, float)):
                        cluster_props[key].append(value)

        return {
            key: np.mean(values) if values else 0.0
            for key, values in cluster_props.items()
        }


# ============================================================================
# MULTI-SPEED GRADIENT SYSTEM
# ============================================================================

class MultiSpeedGradient(nn.Module):
    """
    Multiple learning speeds: fast patterns → slow understanding
    From your vision: [16, 32, 64, 128] dims
    """

    def __init__(self, vocab_size, speeds=[16, 32, 64, 128], dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.speeds = speeds
        self.num_speeds = len(speeds)

        print(f"⚡ Multi-speed gradient: {speeds} dims")

        # Separate embeddings for each speed
        self.speed_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, dim)
            for dim in speeds
        ])

        # Separate output heads for each speed
        self.speed_outputs = nn.ModuleList([
            nn.Linear(dim, vocab_size)
            for dim in speeds
        ])

        # Learned mixing weights (which speeds work best?)
        self.speed_weights = nn.Parameter(torch.ones(self.num_speeds) / self.num_speeds)

        # Per-speed processing
        self.speed_processors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout)
            )
            for dim in speeds
        ])

        # Initialize
        for embed in self.speed_embeddings:
            nn.init.normal_(embed.weight, std=0.02)

        for output in self.speed_outputs:
            nn.init.xavier_uniform_(output.weight)

    def forward(self, x):
        """
        x: [batch, seq] token ids
        Returns: combined logits [batch, vocab_size]
        """
        batch, seq = x.shape

        # Get predictions from each speed
        speed_logits = []

        for i, (embed, processor, output) in enumerate(zip(
            self.speed_embeddings,
            self.speed_processors,
            self.speed_outputs
        )):
            # Embed at this speed
            h = embed(x)  # [batch, seq, speed_dim]

            # Process
            h = processor(h)

            # Get logits (last position only)
            logits = output(h[:, -1, :])  # [batch, vocab_size]

            speed_logits.append(logits)

        # Stack: [num_speeds, batch, vocab_size]
        speed_logits = torch.stack(speed_logits, dim=0)

        # Normalize weights
        weights = F.softmax(self.speed_weights, dim=0)

        # Weighted combination: [batch, vocab_size]
        combined = (speed_logits * weights.view(-1, 1, 1)).sum(dim=0)

        return combined, speed_logits, weights


# ============================================================================
# SUBCONSCIOUS PLANNING SYSTEM (4-LAYER)
# ============================================================================

class SeaOfNoise(nn.Module):
    """
    Layer 0: Continuous activation field for all concepts
    """

    def __init__(self, vocab_size, hidden_dim, noise_dim=32):  # ← ADD hidden_dim!
        super().__init__()

        self.vocab_size = vocab_size
        self.noise_dim = noise_dim

        # Continuous activation values
        self.concept_field = nn.Parameter(torch.randn(vocab_size, noise_dim) * 0.1)

        # Context influence - FIX THE DIMENSIONS!
        self.context_proj = nn.Linear(hidden_dim, noise_dim)  # ← CHANGED: hidden_dim → noise_dim

    def forward(self, context_embedding):
        """
        context_embedding: [batch, hidden_dim]  ← Receives hidden_dim!
        Returns: [batch, vocab_size, noise_dim] activation field
        """
        batch = context_embedding.size(0)

        # Base field (same for all)
        field = self.concept_field.unsqueeze(0).expand(batch, -1, -1)

        # Influence from context
        context_influence = self.context_proj(context_embedding)  # Now works: 128 → 32
        context_influence = context_influence.unsqueeze(1)  # [batch, 1, noise_dim]

        # Modulate field
        field = field + context_influence * 0.3

        return field


class PeakDetector(nn.Module):
    """
    Layer 1: Select peaks (relevant) + random exploration
    """

    def __init__(self, noise_dim, num_peaks=10, exploration_rate=0.1):
        super().__init__()

        self.noise_dim = noise_dim
        self.num_peaks = num_peaks
        self.exploration_rate = exploration_rate

        # Attention for peak selection
        self.attention = nn.Linear(noise_dim, 1)

    def forward(self, activation_field, training=True):
        """
        activation_field: [batch, vocab_size, noise_dim]
        Returns: selected concepts [batch, num_peaks, noise_dim]
        """
        batch, vocab_size, noise_dim = activation_field.shape

        # Compute attention scores
        scores = self.attention(activation_field).squeeze(-1)  # [batch, vocab_size]

        # Select top peaks
        num_peaks = min(self.num_peaks, vocab_size)
        top_k = torch.topk(scores, k=num_peaks, dim=-1)
        peak_indices = top_k.indices  # [batch, num_peaks]

        # Exploration: replace some with random
        if training and random.random() < self.exploration_rate:
            num_explore = max(1, int(num_peaks * 0.2))
            random_indices = torch.randint(0, vocab_size, (batch, num_explore), device=peak_indices.device)
            peak_indices[:, -num_explore:] = random_indices

        # Gather selected concepts
        peak_indices_expanded = peak_indices.unsqueeze(-1).expand(-1, -1, noise_dim)
        selected = torch.gather(activation_field, 1, peak_indices_expanded)

        return selected


class FutureGenerators(nn.Module):
    """
    Layer 2: Multiple generators predict different futures
    """

    def __init__(self, noise_dim, hidden_dim, num_generators=5, predict_ahead=5):
        super().__init__()

        self.num_generators = num_generators
        self.predict_ahead = predict_ahead

        # Small independent generators
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(noise_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, predict_ahead * hidden_dim)
            )
            for _ in range(num_generators)
        ])

    def forward(self, selected_concepts):
        """
        selected_concepts: [batch, num_peaks, noise_dim]
        Returns: futures [num_generators, batch, predict_ahead, hidden_dim]
        """
        batch = selected_concepts.size(0)

        # Average over selected concepts
        context = selected_concepts.mean(dim=1)  # [batch, noise_dim]

        # Generate futures
        futures = []

        for generator in self.generators:
            future = generator(context)  # [batch, predict_ahead * hidden_dim]
            future = future.view(batch, self.predict_ahead, -1)
            futures.append(future)

        return torch.stack(futures, dim=0)


class ScenarioEvaluator(nn.Module):
    """
    Layer 3: Evaluate quality of predicted futures
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Quality scoring network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, futures):
        """
        futures: [num_generators, batch, predict_ahead, hidden_dim]
        Returns: scores [num_generators, batch], best_future [batch, hidden_dim]
        """
        num_gen, batch, ahead, hidden = futures.shape

        # Score each future (average over prediction steps)
        scores = []

        for i in range(num_gen):
            future = futures[i]  # [batch, predict_ahead, hidden_dim]
            future_avg = future.mean(dim=1)  # [batch, hidden_dim]
            score = self.scorer(future_avg).squeeze(-1)  # [batch]
            scores.append(score)

        scores = torch.stack(scores, dim=0)  # [num_generators, batch]

        # Select best future per batch
        best_indices = scores.argmax(dim=0)  # [batch]

        # Gather best futures
        best_futures = []
        for b in range(batch):
            best_idx = best_indices[b].item()
            best_future = futures[best_idx, b, 0, :]  # First predicted step
            best_futures.append(best_future)

        best_futures = torch.stack(best_futures, dim=0)  # [batch, hidden_dim]

        return scores, best_futures


class InfluenceGenerator(nn.Module):
    """
    Layer 4: Convert best future into influence vector
    """

    def __init__(self, hidden_dim, influence_strength=0.3):
        super().__init__()

        self.influence_strength = influence_strength

        # Project to influence space
        self.to_influence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded influence
        )

    def forward(self, best_futures):
        """
        best_futures: [batch, hidden_dim]
        Returns: influence [batch, hidden_dim]
        """
        influence = self.to_influence(best_futures)
        return influence * self.influence_strength


class SubconsciousPlanning(nn.Module):
    """
    Complete 4-layer subconscious system
    """

    def __init__(self, vocab_size, hidden_dim, noise_dim=32):  # ← hidden_dim needed!
        super().__init__()

        print(f"🧠 Subconscious planning: 4-layer predictive system")

        self.layer0 = SeaOfNoise(vocab_size, hidden_dim, noise_dim)  # ← Pass hidden_dim!
        self.layer1 = PeakDetector(noise_dim, num_peaks=10)
        self.layer2 = FutureGenerators(noise_dim, hidden_dim, num_generators=5)
        self.layer3 = ScenarioEvaluator(hidden_dim)
        self.layer4 = InfluenceGenerator(hidden_dim)

    def forward(self, context_embedding, training=True):
        """
        context_embedding: [batch, hidden_dim]
        Returns: influence [batch, hidden_dim]
        """
        # Layer 0: Sea of noise
        field = self.layer0(context_embedding)  # Now works!

        # Layer 1: Peak detection
        selected = self.layer1(field, training=training)

        # Layer 2: Generate futures
        futures = self.layer2(selected)

        # Layer 3: Evaluate scenarios
        scores, best_future = self.layer3(futures)

        # Layer 4: Generate influence
        influence = self.layer4(best_future)

        return influence


# ============================================================================
# ATTENTION-GUIDED DOMAIN EMBEDDINGS
# ============================================================================

class AttentionGuidedEmbedding(nn.Module):
    """
    Embeddings that grow domains based on attention patterns
    """

    def __init__(self, vocab_size, base_dim=8, max_domains=30):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.max_domains = max_domains

        # Microscopic base
        self.base_embed = nn.Embedding(vocab_size, base_dim)
        nn.init.normal_(self.base_embed.weight, std=0.02)

        # Attention-discovered domains
        self.domains = nn.ModuleList()
        self.domain_tokens = []
        self.domain_sizes = []

        # Difficulty tracking
        self.token_losses = defaultdict(float)
        self.token_counts = defaultdict(int)

        print(f"🌿 Attention-guided embeddings: {base_dim} base dims")

    def update_difficulty(self, token_id, loss_value):
        """Track struggling tokens"""
        if token_id >= self.vocab_size:
            return
        self.token_losses[token_id] += loss_value
        self.token_counts[token_id] += 1

    def get_struggling_tokens(self, min_samples=10):
        """Find high-loss tokens"""
        if not self.token_counts:
            return {}

        avg_losses = {}
        for tid, count in self.token_counts.items():
            if count >= min_samples:
                avg_losses[tid] = self.token_losses[tid] / count

        if not avg_losses:
            return {}

        overall_avg = sum(avg_losses.values()) / len(avg_losses)

        return {
            tid: loss for tid, loss in avg_losses.items()
            if loss > overall_avg * 1.2
        }

    def create_domain_from_attention(self, token_cluster, attention_props, domain_size):
        """Create domain guided by attention"""
        if len(self.domains) >= self.max_domains:
            return False

        if len(token_cluster) < 3:
            return False

        device = self.base_embed.weight.device

        # Domain as residual correction
        domain = nn.Sequential(
            nn.Linear(self.base_dim, domain_size, bias=False),
            nn.GELU(),
            nn.Linear(domain_size, self.base_dim, bias=False)
        ).to(device)

        for param in domain.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

        self.domains.append(domain)
        self.domain_tokens.append(set(token_cluster))
        self.domain_sizes.append(domain_size)

        return True

    def expand_base(self, additional_dims):
        """Expand base embeddings (for fundamental concepts)"""
        old_embed = self.base_embed.weight
        old_dim = old_embed.size(1)
        new_dim = old_dim + additional_dims

        device = old_embed.device
        new_embed = nn.Embedding(self.vocab_size, new_dim).to(device)

        with torch.no_grad():
            new_embed.weight[:, :old_dim] = old_embed
            nn.init.normal_(new_embed.weight[:, old_dim:], std=0.02)

        self.base_embed = new_embed
        self.base_dim = new_dim

        return True

    def forward(self, x):
        """
        x: [batch, seq]
        Returns: [batch, seq, base_dim]
        """
        h = self.base_embed(x)

        # Apply domain corrections
        if len(self.domains) > 0:
            batch, seq = x.shape

            for b in range(batch):
                for s in range(seq):
                    tid = x[b, s].item()

                    for domain_idx, token_set in enumerate(self.domain_tokens):
                        if tid in token_set:
                            base_vec = h[b:b+1, s:s+1, :]
                            correction = self.domains[domain_idx](base_vec)
                            h[b, s, :] = h[b, s, :] + correction.squeeze() * 0.1

        return h


# ============================================================================
# TRANSFORMER WITH ATTENTION CAPTURE
# ============================================================================

class AttentionCaptureBlock(nn.Module):
    """
    Transformer block that captures attention patterns
    """

    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.last_attention = None

    def forward(self, x, subconscious_influence=None):
        """
        x: [batch, seq, dim]
        subconscious_influence: [batch, dim] (optional)
        """
        seq_len = x.size(1)

        # Add subconscious influence
        if subconscious_influence is not None:
            x = x + subconscious_influence.unsqueeze(1) * 0.3

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        # Attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False
        )

        self.last_attention = attn_weights.detach()

        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================================
# ATTENTION-ORGANIC GROWTH CONTROLLER
# ============================================================================

class AttentionOrganicGrowth:
    """
    Growth driven by attention patterns
    """

    def __init__(self, model, vocab_size):
        self.model = model
        self.vocab_size = vocab_size

        self.attention_analyzer = AttentionAnalyzer(vocab_size)
        self.growth_events = []
        self.last_growth_check = 0

        print("🌊 Attention-organic growth controller")

    def capture_attention_patterns(self, blocks, token_ids):
        """Capture from all blocks"""
        for block in blocks:
            if hasattr(block, 'last_attention') and block.last_attention is not None:
                self.attention_analyzer.capture_attention(
                    block.last_attention,
                    token_ids
                )

    def check_and_grow(self, current_step):
        """Check for growth based on attention"""
        if current_step - self.last_growth_check < 50:
            return False

        self.last_growth_check = current_step

        if len(self.attention_analyzer.attention_snapshots) < 50:
            return False

        # Compute attention properties
        properties = self.attention_analyzer.compute_attention_properties()

        if not properties:
            return False

        # Get struggling tokens
        struggling = self.model.embed.get_struggling_tokens(min_samples=10)

        if not struggling:
            return False

        # Find clusters among struggling tokens
        struggling_properties = {
            tid: properties[tid]
            for tid in struggling.keys()
            if tid in properties
        }

        clusters = self.attention_analyzer.find_attention_clusters(
            struggling_properties,
            loss_dict=struggling
        )

        if not clusters:
            return False

        # Pick worst cluster
        cluster_losses = [
            (np.mean([struggling.get(tid, 0) for tid in cluster]), cluster)
            for cluster in clusters
        ]

        worst_loss, worst_cluster = max(cluster_losses, key=lambda x: x[0])

        # Get cluster attention properties
        cluster_props = self.attention_analyzer.get_cluster_properties(
            worst_cluster,
            properties
        )

        if not cluster_props:
            return False

        # Determine growth from attention
        growth_decision = self.determine_growth_from_attention(
            worst_cluster,
            cluster_props,
            worst_loss
        )

        if not growth_decision:
            return False

        # Execute
        return self.execute_growth(growth_decision, current_step)

    def determine_growth_from_attention(self, cluster, attention_props, avg_loss):
        """
        Attention properties determine growth
        """
        ratio = attention_props.get('attention_ratio', 1.0)
        stability = attention_props.get('stability', 0.0)
        incoming = attention_props.get('incoming_mean', 0.0)

        # Compute size from attention
        size_score = ratio * 5 + stability * 10 + incoming * 20
        domain_size = int(max(8, min(64, size_score)))
        domain_size = (domain_size // 4) * 4

        # Determine type from attention shape
        if ratio > 1.8 and stability > 0.5:
            growth_type = 'expand_base'
        elif stability > 0.6 and len(cluster) >= 5:
            growth_type = 'create_domain'
        else:
            growth_type = 'small_domain'
            domain_size = min(domain_size, 16)

        return {
            'type': growth_type,
            'cluster': cluster,
            'size': domain_size,
            'attention': attention_props,
            'loss': avg_loss
        }

    def execute_growth(self, decision, step):
        """Execute growth"""
        growth_type = decision['type']
        cluster = decision['cluster']
        size = decision['size']

        if growth_type == 'expand_base':
            print(f"\n  🌍 Expanding BASE +{size//2} dims")
            print(f"     Ratio: {decision['attention']['attention_ratio']:.2f}")
            print(f"     Tokens: {len(cluster)}")

            success = self.model.embed.expand_base(size // 2)

            if success:
                # Update input projection
                device = next(self.model.parameters()).device
                new_proj = nn.Linear(self.model.embed.base_dim, self.model.hidden_dim).to(device)
                self.model.input_proj = new_proj

        else:
            print(f"\n  🌿 Creating domain")
            print(f"     Size: {size} dims")
            print(f"     Tokens: {len(cluster)}")
            print(f"     Stability: {decision['attention']['stability']:.2f}")

            success = self.model.embed.create_domain_from_attention(
                cluster,
                decision['attention'],
                size
            )

        if success:
            self.growth_events.append({
                'step': step,
                'type': growth_type,
                'size': size,
                'cluster_size': len(cluster)
            })

            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"     Total params: {total_params:,}")

        return success


# ============================================================================
# COMPLETE v15 MODEL
# ============================================================================

class ExpandFormerV15Complete(nn.Module):
    """
    COMPLETE v15: All systems integrated
    - Multi-speed gradient learning
    - Subconscious planning
    - Attention-guided embeddings
    - Organic growth
    """

    def __init__(self, vocab_size, base_dim=8, hidden_dim=128,
                 context_len=128, num_blocks=2, num_heads=2,
                 dropout=0.1, max_domains=30):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_domains = max_domains

        print("=" * 70)
        print("🌟 EXPANDFORMER V15 - COMPLETE ASI ARCHITECTURE")
        print("=" * 70)
        print(f"Components:")
        print(f"  ✅ Multi-speed gradient (16, 32, 64, 128 dims)")
        print(f"  ✅ Subconscious planning (4-layer system)")
        print(f"  ✅ Attention-guided embeddings")
        print(f"  ✅ Organic growth controller")
        print(f"\nConfiguration:")
        print(f"  Base: {base_dim} dims")
        print(f"  Hidden: {hidden_dim} dims")
        print(f"  Blocks: {num_blocks}")
        print(f"  Heads: {num_heads}")

        # Multi-speed system
        self.multi_speed = MultiSpeedGradient(
            vocab_size,
            speeds=[16, 32, 64, 128],
            dropout=dropout
        )

        # Subconscious planning
        self.subconscious = SubconsciousPlanning(
            vocab_size,
            hidden_dim,
            noise_dim=32
        )

        # Attention-guided embeddings
        self.embed = AttentionGuidedEmbedding(vocab_size, base_dim, max_domains)

        # Positional encoding
        self.register_buffer('pos_enc', self._create_pos_encoding(context_len, base_dim))

        # Project to hidden
        self.input_proj = nn.Linear(base_dim, hidden_dim)

        # Transformer blocks with attention capture
        self.blocks = nn.ModuleList([
            AttentionCaptureBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Growth controller
        self.growth = AttentionOrganicGrowth(self, vocab_size)

        self.total_updates = 0

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n✓ Initial parameters: {total_params:,}")
        print("=" * 70 + "\n")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _create_pos_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x, use_subconscious=True, use_multispeed=True):
        """
        x: [batch, seq]
        Returns: logits [batch, vocab_size]
        """
        batch, seq = x.shape

        # Get multi-speed predictions
        if use_multispeed:
            fast_logits, speed_logits_all, speed_weights = self.multi_speed(x)
        else:
            fast_logits = None

        # Main path through domain embeddings
        h = self.embed(x)  # [batch, seq, base_dim]

        # Positional encoding
        if seq <= self.context_len:
            if h.size(2) == self.pos_enc.size(1):
                h = h + self.pos_enc[:seq].unsqueeze(0)
            else:
                self.register_buffer('pos_enc', self._create_pos_encoding(self.context_len, h.size(2)))
                h = h + self.pos_enc[:seq].unsqueeze(0)

        # Project to hidden
        h = self.input_proj(h)  # [batch, seq, hidden_dim]

        # Get subconscious influence
        if use_subconscious:
            context = h[:, -1, :]  # Last position as context
            subconscious_influence = self.subconscious(context, training=self.training)
        else:
            subconscious_influence = None

        # Transform through blocks (with subconscious influence)
        for block in self.blocks:
            h = block(h, subconscious_influence)

        # Output
        h = self.output_norm(h)
        slow_logits = self.output(h[:, -1, :])  # [batch, vocab_size]

        # Capture attention for growth
        if self.training:
            self.growth.capture_attention_patterns(self.blocks, x)

        # Combine fast and slow
        if use_multispeed and fast_logits is not None:
            # Weight: 30% fast, 70% slow
            combined = 0.3 * fast_logits + 0.7 * slow_logits
        else:
            combined = slow_logits

        return combined

    def update_tracking(self, loss_value, token_id):
        """Update for growth"""
        self.embed.update_difficulty(token_id, loss_value)
        self.total_updates += 1

    def check_and_grow(self):
        """Check attention-based growth"""
        return self.growth.check_and_grow(self.total_updates)


# ============================================================================
# TRAINER
# ============================================================================

class CompleteV15Trainer:
    """
    Trainer for complete v15
    """

    def __init__(self, model, tokenizer, lr=0.0003, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        print("🎓 Complete v15 trainer initialized\n")

    def train(self, texts, epochs=5, context_len=128, sample_every=100):
        """
        Train complete system
        """
        print("🌟 TRAINING: Complete ASI Architecture\n")

        total_updates = 0

        for epoch in range(epochs):
            print(f"{'=' * 70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 70}\n")

            epoch_losses = []
            last_sample = 0

            for text_idx, text in enumerate(texts):
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

                    # Forward
                    self.optimizer.zero_grad()
                    logits = self.model(x)

                    loss = F.cross_entropy(logits, y)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # Track
                    epoch_losses.append(loss.item())
                    self.model.update_tracking(loss.item(), y_token)
                    total_updates += 1

                    # Check growth
                    if total_updates % 50 == 0:
                        self.model.check_and_grow()

                    # Sample
                    if total_updates - last_sample >= sample_every:
                        last_sample = total_updates
                        avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)

                        print(f"\n{'─' * 70}")
                        print(f"Update {total_updates:5d} | Loss: {avg_loss:.4f}")
                        print(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
                        print(f"Domains: {len(self.model.embed.domains)} | "
                              f"Blocks: {len(self.model.blocks)} | "
                              f"Base: {self.model.embed.base_dim}d")

                        # Speed weights
                        weights = F.softmax(self.model.multi_speed.speed_weights, dim=0)
                        speed_str = " | ".join([f"{w:.2f}" for w in weights])
                        print(f"Speed weights: [{speed_str}]")
                        print(f"{'─' * 70}\n")

                if (text_idx + 1) % 20 == 0:
                    avg_loss = np.mean(epoch_losses)
                    print(f"  Progress: {text_idx + 1}/{len(texts)} | loss={avg_loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"\n✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}\n")

        # Final stats
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Growth Summary:")
        print(f"  Domains: {len(self.model.embed.domains)}")
        print(f"  Final params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Growth events: {len(self.model.growth.growth_events)}")

        if self.model.growth.growth_events:
            print(f"\nGrowth Timeline:")
            for event in self.model.growth.growth_events:
                print(f"  {event['step']:4d}: {event['type']} "
                      f"({event['cluster_size']} tokens, {event['size']}d)")

        print("=" * 70 + "\n")

    def save(self, name, save_dir='checkpoints_v15'):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'growth_events': self.model.growth.growth_events,
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {save_path / name}.pt")


# ============================================================================
# MAIN
# ============================================================================

def train_pipeline():
    print("=" * 70)
    print("🌟 ExpandFormer v15 COMPLETE: The Full ASI Vision")
    print("=" * 70)
    print("\nAll systems integrated:")
    print("  • Multi-speed gradient learning")
    print("  • Subconscious planning")
    print("  • Attention hierarchy discovery")
    print("  • Organic domain growth\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load data
    training_dir = Path("training_data")
    training_texts = []

    if training_dir.exists():
        print("📂 Loading training data...")
        for file_path in training_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    lines = [line.strip() for line in text.split('\n') if line.strip()]

                    for i in range(0, len(lines), 6):
                        chunk = '\n'.join(lines[i:i + 6])
                        if len(chunk) > 10:
                            training_texts.append(chunk)

                print(f"   ✓ {file_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

    if not training_texts:
        print("⚠️  No training data, using demo...")
        training_texts = [
            "Hello, how are you? I am doing well.",
            "The sky is blue. The grass is green.",
            "What is your name? My name is Claude.",
            "I enjoy reading books about science.",
        ] * 10

    print(f"\n✓ Loaded {len(training_texts)} chunks\n")

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab: {vocab_size:,}\n")

    # Create complete model
    model = ExpandFormerV15Complete(
        vocab_size=vocab_size,
        base_dim=8,
        hidden_dim=128,
        context_len=128,
        num_blocks=2,
        num_heads=2,
        dropout=0.1,
        max_domains=30
    )

    # Train
    trainer = CompleteV15Trainer(model, tokenizer, lr=0.0003, device=device)
    trainer.train(training_texts, epochs=5, sample_every=100)

    # Save
    trainer.save("final")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print(__doc__)
        elif sys.argv[1] == '--benchmark':
            print("Run: python benchmark_testing.py --quick --versions v15")
        else:
            print(f"Unknown: {sys.argv[1]}")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()

ExpandFormer = ExpandFormerV15Complete