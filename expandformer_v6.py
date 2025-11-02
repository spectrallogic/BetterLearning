"""
ExpandFormer v6: Flash Learning + Diffusion + Adaptive Growth
==============================================================

REVOLUTIONARY FEATURES:
✓ Flash Learning: Instant global structure from corpus scan
✓ Diffusion Passes: Multi-scale learning (coarse → fine)
✓ Semantic Growth: Blocks split based on semantic clustering
✓ Attention Memory: Memory hierarchy from attention patterns
✓ Real-time Learning: Continue learning after initialization
✓ Standard Tokenization: GPT-2 style (tiktoken)

ARCHITECTURE PHILOSOPHY:
- Flash scan discovers semantic structure
- Blocks specialize in semantic clusters
- Growth is strategic, not reactive
- Memory emerges from attention patterns
- Learning happens at multiple scales

REQUIREMENTS:
pip install torch tiktoken numpy scikit-learn

USAGE:
python expandformer_v6.py              # Full training pipeline
python expandformer_v6.py --chat       # Chat mode
python expandformer_v6.py --analyze    # Analyze saved model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict, Counter
import threading
from pathlib import Path
from datetime import datetime
import sys
import json

try:
    import tiktoken
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install tiktoken scikit-learn")
    sys.exit(1)


# ============================================================================
# PHASE 1: FLASH LEARNING COMPONENTS
# ============================================================================

class FlashScanner:
    """
    Phase 1: Global corpus analysis
    Scans entire corpus to discover structure before any training
    """

    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        # Co-occurrence: which tokens appear near each other
        self.co_occurrence = defaultdict(Counter)

        # Token frequency
        self.token_freq = Counter()

        # Token positions in corpus
        self.token_positions = defaultdict(list)

        # Computed metrics
        self.centrality_scores = None
        self.semantic_clusters = None

    def scan_corpus(self, texts, window=10):
        """
        FLASH SCAN: Analyze entire corpus structure instantly

        This is like speed-reading to get the gist before deep reading
        """
        print("\n" + "=" * 70)
        print("⚡ PHASE 1: FLASH SCAN")
        print("=" * 70)
        print("\nScanning corpus structure...")

        all_tokens = []

        # Collect all tokens with positions
        for text_idx, text in enumerate(texts):
            tokens = self.tokenizer.encode(text)

            for pos, token in enumerate(tokens):
                global_pos = len(all_tokens) + pos
                self.token_positions[token].append(global_pos)
                self.token_freq[token] += 1

            all_tokens.extend(tokens)

        print(f"✓ Scanned {len(all_tokens):,} tokens")
        print(f"✓ Found {len(self.token_positions)} unique tokens")

        # Build co-occurrence graph
        print("\nBuilding co-occurrence graph...")
        for i, token in enumerate(all_tokens):
            if i % 10000 == 0:
                print(f"  Progress: {i:,}/{len(all_tokens):,}", end='\r')

            context_start = max(0, i - window)
            context_end = min(len(all_tokens), i + window + 1)

            for j in range(context_start, context_end):
                if i != j:
                    context_token = all_tokens[j]
                    distance = abs(i - j)
                    weight = 1.0 / distance  # Closer = stronger connection
                    self.co_occurrence[token][context_token] += weight

        print(f"\n✓ Built co-occurrence graph")

        # Compute centrality
        self._compute_centrality()

        # Discover semantic clusters
        self._discover_semantic_clusters(all_tokens)

        return self.get_flash_summary()

    def _compute_centrality(self):
        """
        Compute attention centrality for each token
        High centrality = abstract, low = episodic
        """
        print("\nComputing token centrality...")

        centrality = np.zeros(self.vocab_size)

        for token_id in self.co_occurrence.keys():
            if token_id >= self.vocab_size:
                continue

            # In-degree: how many tokens connect TO this one
            in_degree = len(self.co_occurrence[token_id])

            # Weighted connections
            total_weight = sum(self.co_occurrence[token_id].values())

            # Frequency
            frequency = self.token_freq[token_id]

            # Combined score
            centrality[token_id] = (
                    0.5 * (in_degree / 100) +  # Normalize connections
                    0.3 * (total_weight / 1000) +  # Normalize weights
                    0.2 * (frequency / 1000)  # Normalize frequency
            )

        # Normalize to [0, 1]
        max_centrality = centrality.max()
        if max_centrality > 0:
            centrality = centrality / max_centrality

        self.centrality_scores = torch.from_numpy(centrality).float()

        print("✓ Centrality computed")

    def _discover_semantic_clusters(self, all_tokens, num_clusters=8):
        """
        Discover semantic clusters in the corpus
        These will guide block specialization
        """
        print(f"\nDiscovering {num_clusters} semantic clusters...")

        # Build token context vectors
        token_contexts = {}

        for token_id in self.co_occurrence.keys():
            if token_id >= self.vocab_size:
                continue

            # Create a sparse vector of co-occurrences
            context_vec = np.zeros(min(1000, self.vocab_size))  # Limit size

            for co_token, weight in self.co_occurrence[token_id].most_common(1000):
                if co_token < len(context_vec):
                    context_vec[co_token] = weight

            # Normalize
            norm = np.linalg.norm(context_vec)
            if norm > 0:
                context_vec = context_vec / norm

            token_contexts[token_id] = context_vec

        # Cluster tokens by context similarity
        if len(token_contexts) >= num_clusters:
            token_ids = list(token_contexts.keys())
            context_matrix = np.array([token_contexts[tid] for tid in token_ids])

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(context_matrix)

            # Store cluster assignments
            self.semantic_clusters = {}
            for token_id, cluster_id in zip(token_ids, cluster_labels):
                self.semantic_clusters[token_id] = int(cluster_id)

            # Analyze clusters
            cluster_sizes = Counter(cluster_labels)
            print(f"✓ Clusters discovered:")
            for cluster_id in range(num_clusters):
                size = cluster_sizes[cluster_id]
                # Get sample tokens from this cluster
                sample_tokens = [
                    tid for tid, cid in self.semantic_clusters.items()
                    if cid == cluster_id
                ][:5]
                sample_words = [self.tokenizer.decode([tid]) for tid in sample_tokens]
                print(f"  Cluster {cluster_id}: {size} tokens (e.g., {sample_words})")
        else:
            print("⚠️  Not enough tokens for clustering")
            self.semantic_clusters = {}

    def get_flash_summary(self):
        """Get summary of flash scan results"""
        if self.centrality_scores is None:
            return {}

        # Top abstract tokens
        top_k = 10
        top_centrality, top_indices = torch.topk(self.centrality_scores, min(top_k, (self.centrality_scores > 0).sum()))

        abstract_tokens = []
        for idx, score in zip(top_indices, top_centrality):
            try:
                token_text = self.tokenizer.decode([idx.item()])
                abstract_tokens.append((token_text, score.item()))
            except:
                pass

        # Bottom episodic tokens
        valid_scores = self.centrality_scores[self.centrality_scores > 0]
        if len(valid_scores) > 0:
            bottom_k = min(top_k, len(valid_scores))
            bottom_centrality, bottom_indices = torch.topk(
                self.centrality_scores,
                bottom_k,
                largest=False
            )

            episodic_tokens = []
            for idx, score in zip(bottom_indices, bottom_centrality):
                if score > 0:  # Only non-zero
                    try:
                        token_text = self.tokenizer.decode([idx.item()])
                        episodic_tokens.append((token_text, score.item()))
                    except:
                        pass
        else:
            episodic_tokens = []

        return {
            'abstract_tokens': abstract_tokens,
            'episodic_tokens': episodic_tokens,
            'num_clusters': len(set(self.semantic_clusters.values())) if self.semantic_clusters else 0,
            'total_tokens': len(self.token_positions),
        }

    def get_token_cluster(self, token_id):
        """Get semantic cluster for a token"""
        return self.semantic_clusters.get(token_id, 0)


# ============================================================================
# ATTENTION & MEMORY COMPONENTS
# ============================================================================

class AttentionCentralityTracker:
    """Tracks attention patterns to compute memory hierarchy"""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.attention_received = torch.zeros(vocab_size)
        self.attention_given = torch.zeros(vocab_size)
        self.token_counts = torch.zeros(vocab_size)
        self.centrality_scores = torch.zeros(vocab_size)

    def update(self, token_ids, attention_weights):
        """Update from attention patterns"""
        avg_attention = attention_weights.mean(dim=0)
        seq_len = token_ids.shape[0]

        for i in range(seq_len):
            token_id = token_ids[i].item()
            if token_id >= self.vocab_size:
                continue

            received = avg_attention[:, i].sum().item()
            given = avg_attention[i, :].sum().item()

            self.attention_received[token_id] += received
            self.attention_given[token_id] += given
            self.token_counts[token_id] += 1

    def compute_centrality(self):
        """Compute centrality scores"""
        mask = self.token_counts > 0

        normalized_received = torch.zeros_like(self.attention_received)
        normalized_given = torch.zeros_like(self.attention_given)

        normalized_received[mask] = self.attention_received[mask] / self.token_counts[mask]
        normalized_given[mask] = self.attention_given[mask] / self.token_counts[mask]

        self.centrality_scores = 0.7 * normalized_received + 0.3 * normalized_given

        max_score = self.centrality_scores.max()
        if max_score > 0:
            self.centrality_scores = self.centrality_scores / max_score

        return self.centrality_scores


class MemoryAwareAttention(nn.Module):
    """Self-attention with memory tracking"""

    def __init__(self, hidden_dim, num_heads, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.centrality_tracker = AttentionCentralityTracker(vocab_size)

    def forward(self, x, token_ids=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        if self.training and token_ids is not None:
            self.centrality_tracker.update(token_ids[0], attn_weights[0].detach().cpu())

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)

        return output


class HierarchicalEmbedding(nn.Module):
    """Embedding with per-token learning rates"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.register_buffer('lr_multipliers', torch.ones(vocab_size))

    def forward(self, token_ids):
        return self.embedding(token_ids)

    def initialize_from_flash(self, centrality_scores):
        """Initialize learning rates from flash scan"""
        # Abstract tokens (high centrality) learn slowly
        self.lr_multipliers = 1.0 - 0.9 * centrality_scores
        self.lr_multipliers = torch.clamp(self.lr_multipliers, 0.05, 1.0)

    def update_lr_multipliers(self, centrality_scores):
        """Update learning rates (blend with current)"""
        new_multipliers = 1.0 - 0.9 * centrality_scores
        new_multipliers = torch.clamp(new_multipliers, 0.05, 1.0)

        # Blend: 80% old, 20% new (smooth adaptation)
        self.lr_multipliers = 0.8 * self.lr_multipliers + 0.2 * new_multipliers


# ============================================================================
# ADAPTIVE GROWTH COMPONENTS
# ============================================================================

class SemanticSpecializedBlock(nn.Module):
    """
    Attention block specialized for a semantic cluster
    Can split if cluster becomes too complex
    """

    def __init__(self, hidden_dim, num_heads, vocab_size, block_id="B0",
                 semantic_cluster=None, specialization_score=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.block_id = block_id

        # Semantic specialization
        self.semantic_cluster = semantic_cluster  # Which cluster does this handle?
        self.specialization_score = specialization_score  # How specialized?

        # Core components
        self.attention = MemoryAwareAttention(hidden_dim, num_heads, vocab_size)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Splitting mechanism
        self.is_split = False
        self.child_blocks = nn.ModuleList()

        # Performance tracking
        self.register_buffer('recent_losses', torch.zeros(20))
        self.loss_idx = 0
        self.register_buffer('avg_loss', torch.tensor(0.0))
        self.updates_since_birth = 0

        # Cluster load tracking (for smart splitting)
        self.cluster_token_counts = Counter()

    def forward(self, x, token_ids=None):
        if self.is_split and len(self.child_blocks) > 0:
            # Route through children
            outputs = []
            for child in self.child_blocks:
                out = child(x, token_ids)
                outputs.append(out)
            return torch.stack(outputs).mean(dim=0)

        attended = self.attention(x, token_ids)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        return x

    def update_performance(self, loss_value, token_ids=None):
        """Track performance and cluster load"""
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 20
        self.avg_loss = self.recent_losses.mean()
        self.updates_since_birth += 1

        # Track which tokens from our cluster we're seeing
        if token_ids is not None and self.semantic_cluster is not None:
            for tid in token_ids[0].tolist():
                self.cluster_token_counts[tid] += 1

    def should_split(self, flash_scanner, min_updates=50, loss_threshold=1.5):
        """
        Decide if this block should split

        Strategy: Split if handling too diverse a semantic cluster
        """
        if self.is_split or self.updates_since_birth < min_updates:
            return False

        # Check 1: High loss (confused)
        if self.avg_loss > loss_threshold:
            return True

        # Check 2: Handling too many diverse tokens from cluster
        if len(self.cluster_token_counts) > 100:  # Arbitrary threshold
            # Check if they're diverse
            top_tokens = [tid for tid, _ in self.cluster_token_counts.most_common(50)]

            # If these tokens have low co-occurrence with each other, split
            if flash_scanner and flash_scanner.co_occurrence:
                diversity = self._compute_token_diversity(top_tokens, flash_scanner)
                if diversity > 0.7:  # High diversity = need split
                    return True

        return False

    def _compute_token_diversity(self, token_ids, flash_scanner):
        """Measure how diverse a set of tokens is"""
        if len(token_ids) < 2:
            return 0.0

        # Check co-occurrence between tokens
        total_pairs = 0
        connected_pairs = 0

        for i, t1 in enumerate(token_ids[:20]):  # Sample
            for t2 in token_ids[i + 1:20]:
                total_pairs += 1
                if t2 in flash_scanner.co_occurrence.get(t1, {}):
                    connected_pairs += 1

        if total_pairs == 0:
            return 0.0

        # Low connection rate = high diversity
        connection_rate = connected_pairs / total_pairs
        diversity = 1.0 - connection_rate

        return diversity

    def split(self, flash_scanner, num_children=2):
        """Split into specialized children"""
        if self.is_split:
            return False

        print(
            f"  🌱 Splitting {self.block_id}: loss={self.avg_loss:.3f}, specialization={self.specialization_score:.2f}")

        device = next(self.parameters()).device

        # Analyze our token distribution to create sub-specializations
        top_tokens = [tid for tid, _ in self.cluster_token_counts.most_common(100)]

        # Cluster these tokens into sub-groups
        if len(top_tokens) >= num_children and flash_scanner:
            sub_clusters = self._create_sub_specializations(top_tokens, flash_scanner, num_children)
        else:
            sub_clusters = [None] * num_children

        for i in range(num_children):
            child = SemanticSpecializedBlock(
                self.hidden_dim,
                self.num_heads,
                self.vocab_size,
                block_id=f"{self.block_id}.{i}",
                semantic_cluster=sub_clusters[i],
                specialization_score=self.specialization_score * 1.2
            ).to(device)

            # Inherit weights
            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        noise = torch.randn_like(parent_param) * 0.01
                        child_param.data.copy_(parent_param.data + noise)

            self.child_blocks.append(child)

        self.is_split = True

        # Freeze parent
        for param in self.parameters():
            param.requires_grad = False

        # Activate children
        for child in self.child_blocks:
            for param in child.parameters():
                param.requires_grad = True

        return True

    def _create_sub_specializations(self, token_ids, flash_scanner, num_groups):
        """Divide tokens into sub-specialization groups"""
        # Simple clustering by frequency
        tokens_per_group = len(token_ids) // num_groups

        sub_clusters = []
        for i in range(num_groups):
            start = i * tokens_per_group
            end = start + tokens_per_group if i < num_groups - 1 else len(token_ids)
            sub_clusters.append(set(token_ids[start:end]))

        return sub_clusters


# ============================================================================
# MAIN MODEL
# ============================================================================

class FlashDiffusionTransformer(nn.Module):
    """
    Main model: Combines flash learning, diffusion, and adaptive growth
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 context_len=128, num_initial_blocks=4, num_heads=4):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # Hierarchical embedding
        self.embedding = HierarchicalEmbedding(vocab_size, embed_dim)

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, embed_dim))

        # Projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Semantic specialized blocks
        self.blocks = nn.ModuleList([
            SemanticSpecializedBlock(
                hidden_dim, num_heads, vocab_size,
                block_id=f"B{i}",
                semantic_cluster=i,  # Initially one cluster per block
                specialization_score=0.5
            )
            for i in range(num_initial_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Stats
        self.total_updates = 0
        self.total_splits = 0
        self.tokens_learned = 0

        # Flash scan results (will be set during initialization)
        self.flash_scanner = None

    def _create_pos_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def initialize_from_flash(self, flash_scanner):
        """Initialize model from flash scan results"""
        print("\n⚡ Initializing model from flash scan...")

        self.flash_scanner = flash_scanner

        # Initialize embedding learning rates
        self.embedding.initialize_from_flash(flash_scanner.centrality_scores)

        # Assign semantic clusters to blocks
        if flash_scanner.semantic_clusters:
            num_clusters = len(set(flash_scanner.semantic_clusters.values()))
            clusters_per_block = max(1, num_clusters // len(self.blocks))

            for i, block in enumerate(self.blocks):
                block.semantic_cluster = i * clusters_per_block
                block.specialization_score = 0.5

        print("✓ Model initialized from flash scan")

    def forward(self, x, track_attention=False):
        batch_size, seq_len = x.shape

        token_ids = x if track_attention else None

        h = self.embedding(x)
        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h, token_ids)

        h = self.output_norm(h)
        logits = self.output(h)

        return logits

    def get_all_blocks(self):
        """Get all active blocks (including children)"""

        def get_leaves(block):
            if block.is_split and len(block.child_blocks) > 0:
                leaves = []
                for child in block.child_blocks:
                    leaves.extend(get_leaves(child))
                return leaves
            else:
                return [block]

        all_leaves = []
        for block in self.blocks:
            all_leaves.extend(get_leaves(block))
        return all_leaves

    def check_and_split(self):
        """Check if any blocks should split"""
        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(self.flash_scanner, min_updates=50, loss_threshold=1.5):
                if block.split(self.flash_scanner):
                    self.total_splits += 1
                    return True

        return False

    def update_memory_hierarchy(self):
        """Update per-token learning rates from attention patterns"""
        all_centrality = torch.zeros(self.vocab_size)
        num_trackers = 0

        for block in self.get_all_blocks():
            centrality = block.attention.centrality_tracker.compute_centrality()
            all_centrality += centrality
            num_trackers += 1

        if num_trackers > 0:
            all_centrality /= num_trackers
            self.embedding.update_lr_multipliers(all_centrality)

    def save_checkpoint(self, path):
        """Save model"""
        checkpoint = {
            'model_state': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'context_len': self.context_len,
                'num_initial_blocks': len(self.blocks),
            },
            'stats': {
                'total_updates': self.total_updates,
                'total_splits': self.total_splits,
                'tokens_learned': self.tokens_learned,
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'])
        self.total_updates = checkpoint['stats']['total_updates']
        self.total_splits = checkpoint['stats']['total_splits']
        self.tokens_learned = checkpoint['stats']['tokens_learned']


# ============================================================================
# DIFFUSION LEARNING SYSTEM
# ============================================================================

class FlashDiffusionLearner:
    """
    Orchestrates the complete learning pipeline:
    1. Flash scan
    2. Diffusion passes (coarse → fine)
    3. Real-time learning
    """

    def __init__(self, model, tokenizer, flash_scanner, lr=0.003, device='cuda', save_dir='checkpoints_v6'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.flash_scanner = flash_scanner
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.base_lr = lr
        self.optimizer = None  # Created per diffusion pass

        self.running = True
        self.tokens_learned = 0
        self.start_time = time.time()

    def diffusion_pass(self, texts, pass_num, total_passes, focus="contextual"):
        """
        One diffusion pass at a specific granularity

        Args:
            pass_num: Current pass (1, 2, 3, ...)
            total_passes: Total number of passes
            focus: "abstract", "contextual", or "episodic"
        """
        print(f"\n" + "=" * 70)
        print(f"🌊 DIFFUSION PASS {pass_num}/{total_passes}: {focus.upper()}")
        print("=" * 70)

        # Configure pass parameters
        if focus == "abstract":
            lr = self.base_lr * 2.0  # Aggressive for coarse features
            context_window = 64
            sample_ratio = 0.3  # Sample 30% of positions

        elif focus == "contextual":
            lr = self.base_lr
            context_window = 32
            sample_ratio = 0.5

        else:  # episodic
            lr = self.base_lr * 0.5  # Gentle for fine details
            context_window = 16
            sample_ratio = 0.8

        print(f"Config: LR={lr:.4f}, Window={context_window}, Sample={sample_ratio:.0%}")

        # Create optimizer for this pass
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        total_loss = 0
        num_updates = 0

        # Process each text
        for text_idx, text in enumerate(texts):
            token_ids = self.tokenizer.encode(text)

            if len(token_ids) < 2:
                continue

            # DIFFUSION KEY: Random sampling, not sequential
            indices = list(range(len(token_ids) - 1))
            np.random.shuffle(indices)

            # Sample based on pass focus
            num_samples = int(len(indices) * sample_ratio)
            sampled_indices = indices[:num_samples]

            for i in sampled_indices:
                # Get context
                context_start = max(0, i - context_window + 1)
                context = token_ids[context_start:i + 1]

                # Pad
                while len(context) < context_window:
                    context = [0] + context

                x = torch.tensor([context[-context_window:]], dtype=torch.long, device=self.device)
                y = torch.tensor([token_ids[i + 1]], dtype=torch.long, device=self.device)

                # Forward
                self.optimizer.zero_grad()
                logits = self.model(x, track_attention=True)
                loss = F.cross_entropy(logits[:, -1, :], y)

                # FOCUS-BASED WEIGHTING
                if self.flash_scanner and self.flash_scanner.centrality_scores is not None:
                    target_token = token_ids[i + 1]
                    if target_token < len(self.flash_scanner.centrality_scores):
                        centrality = self.flash_scanner.centrality_scores[target_token].item()

                        if focus == "abstract":
                            weight = centrality  # Focus on central tokens
                        elif focus == "episodic":
                            weight = 1.0 - centrality  # Focus on peripheral
                        else:
                            weight = 1.0  # Balanced

                        loss = loss * weight

                loss.backward()

                # Scale embedding gradients by per-token LR
                with torch.no_grad():
                    emb_weight = self.model.embedding.embedding.weight
                    if emb_weight.grad is not None:
                        for tid in range(min(self.model.vocab_size, emb_weight.grad.shape[0])):
                            lr_mult = self.model.embedding.lr_multipliers[tid]
                            emb_weight.grad[tid] *= lr_mult

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update block performance
                for block in self.model.get_all_blocks():
                    block.update_performance(loss.item(), x)

                total_loss += loss.item()
                num_updates += 1
                self.model.total_updates += 1
                self.tokens_learned += 1

            # Progress
            if (text_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_updates if num_updates > 0 else 0
                print(f"  Text {text_idx + 1}/{len(texts)}: loss={avg_loss:.4f}, updates={num_updates}")

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        print(f"\n✓ Pass complete: avg_loss={avg_loss:.4f}")

        # Check for splits after each pass
        if self.model.check_and_split():
            print(f"  🌱 Model grew! Now {len(self.model.get_all_blocks())} blocks")

        return avg_loss

    def train_flash_diffusion(self, texts, num_passes=3):
        """
        Complete training pipeline:
        1. Flash scan (already done)
        2. Multi-pass diffusion
        3. Ready for real-time
        """
        print("\n" + "=" * 70)
        print("PHASE 2: DIFFUSION TRAINING")
        print("=" * 70)

        # Diffusion passes: coarse to fine
        passes = [
            ("abstract", 1),
            ("contextual", 2),
            ("episodic", 3),
        ]

        for focus, pass_num in passes[:num_passes]:
            loss = self.diffusion_pass(texts, pass_num, num_passes, focus=focus)

            # Update memory hierarchy after each pass
            self.model.update_memory_hierarchy()

            # Save checkpoint after each pass
            self.save_checkpoint(f"after_pass{pass_num}_{focus}")

        print("\n✓ Diffusion training complete!")

    def learn_realtime(self, text, show_stats=False):
        """Real-time learning (post-diffusion)"""
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.base_lr * 0.5,  # Gentler for real-time
                weight_decay=0.01
            )

        token_ids = self.tokenizer.encode(text)

        if len(token_ids) < 2:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        for i in range(len(token_ids) - 1):
            context_start = max(0, i - self.model.context_len + 1)
            context = token_ids[context_start:i + 1]

            while len(context) < self.model.context_len:
                context = [0] + context

            x = torch.tensor([context[-self.model.context_len:]], dtype=torch.long, device=self.device)
            y = torch.tensor([token_ids[i + 1]], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, track_attention=True)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()

            # Scale embedding grads
            with torch.no_grad():
                emb_weight = self.model.embedding.embedding.weight
                if emb_weight.grad is not None:
                    for tid in range(min(self.model.vocab_size, emb_weight.grad.shape[0])):
                        lr_mult = self.model.embedding.lr_multipliers[tid]
                        emb_weight.grad[tid] *= lr_mult

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_updates += 1
            self.model.total_updates += 1
            self.tokens_learned += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0

        # Periodic updates
        if self.model.total_updates % 100 == 0:
            self.model.update_memory_hierarchy()

        if self.model.total_updates % 200 == 0:
            self.model.check_and_split()

        return avg_loss

    def generate(self, prompt, max_length=30, temperature=0.8):
        """Generate text"""
        self.model.eval()

        token_ids = self.tokenizer.encode(prompt)

        for _ in range(max_length):
            context = token_ids[-self.model.context_len:]
            while len(context) < self.model.context_len:
                context = [0] + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            token_ids.append(next_token)

            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded in ['.', '!', '?', '\n'] and np.random.random() < 0.3:
                    break
            except:
                pass

        self.model.train()

        return self.tokenizer.decode(token_ids)

    def save_checkpoint(self, name):
        """Save checkpoint"""
        model_path = self.save_dir / f"{name}_model.pt"
        self.model.save_checkpoint(model_path)

        learner_state = {
            'tokens_learned': self.tokens_learned,
            'start_time': self.start_time,
            'lr': self.base_lr,
        }
        learner_path = self.save_dir / f"{name}_learner.json"
        with open(learner_path, 'w') as f:
            json.dump(learner_state, f, indent=2)

        print(f"💾 Saved: {name}")

    @classmethod
    def load_checkpoint(cls, checkpoint_name, device='cuda'):
        """Load checkpoint"""
        save_dir = Path('checkpoints_v6')

        # Load model
        model_path = save_dir / f"{checkpoint_name}_model.pt"
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']

        tokenizer = tiktoken.get_encoding("gpt2")

        model = FlashDiffusionTransformer(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            context_len=config['context_len'],
            num_initial_blocks=config['num_initial_blocks'],
        )
        model.load_checkpoint(model_path)

        # Load learner
        learner_path = save_dir / f"{checkpoint_name}_learner.json"
        with open(learner_path, 'r') as f:
            learner_state = json.load(f)

        # Create dummy flash scanner (won't be used in inference)
        flash_scanner = FlashScanner(tokenizer, config['vocab_size'])

        learner = cls(model, tokenizer, flash_scanner, lr=learner_state['lr'], device=device)
        learner.tokens_learned = learner_state['tokens_learned']
        learner.start_time = learner_state['start_time']

        print(f"✅ Loaded: {checkpoint_name}")
        return learner


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_pipeline():
    """Complete training pipeline"""
    print("=" * 70)
    print("🚀 ExpandFormer v6: Flash + Diffusion + Growth")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load training data
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
                        if chunk:
                            training_texts.append(chunk)
                print(f"   ✓ {file_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

    if not training_texts:
        print("⚠️  No training data, using samples...")
        training_texts = [
            "Hello, how are you? I am doing well.",
            "The sky is blue. The grass is green.",
            "What is your name? My name is Claude.",
        ]

    print(f"\n✓ Loaded {len(training_texts)} text chunks\n")

    # Create tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab size: {vocab_size:,}\n")

    # PHASE 1: FLASH SCAN
    flash_scanner = FlashScanner(tokenizer, vocab_size)
    flash_summary = flash_scanner.scan_corpus(training_texts, window=10)

    print(f"\n📊 Flash Scan Summary:")
    print(f"   Unique tokens: {flash_summary['total_tokens']}")
    print(f"   Semantic clusters: {flash_summary['num_clusters']}")
    print(f"\n   Top Abstract:")
    for token, score in flash_summary['abstract_tokens'][:5]:
        print(f"      '{token}': {score:.3f}")
    print(f"\n   Top Episodic:")
    for token, score in flash_summary['episodic_tokens'][:5]:
        print(f"      '{token}': {score:.3f}")

    # Create model
    print(f"\n🧠 Creating model...")
    model = FlashDiffusionTransformer(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        context_len=128,
        num_initial_blocks=4,
        num_heads=4,
    )
    print(f"✓ {sum(p.numel() for p in model.parameters()):,} parameters")

    # Initialize from flash
    model.initialize_from_flash(flash_scanner)

    # Create learner
    learner = FlashDiffusionLearner(
        model, tokenizer, flash_scanner,
        lr=0.003, device=device
    )

    # PHASE 2: DIFFUSION TRAINING
    learner.train_flash_diffusion(training_texts, num_passes=3)

    # Save final model
    learner.save_checkpoint("final")

    # PHASE 3: REAL-TIME DEMO
    print("\n" + "=" * 70)
    print("PHASE 3: REAL-TIME LEARNING")
    print("=" * 70)
    print("\nType messages to continue training (Ctrl+C to exit)\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            loss = learner.learn_realtime(user_input)
            response = learner.generate(user_input, max_length=30)

            print(f"AI: {response}")
            print(f"   [Loss: {loss:.3f}, Blocks: {len(model.get_all_blocks())}]\n")

    except KeyboardInterrupt:
        print("\n\n✅ Training complete!")
        learner.save_checkpoint("final_interactive")

        print(f"\n📊 Final Stats:")
        print(f"   Blocks: {len(model.get_all_blocks())}")
        print(f"   Splits: {model.total_splits}")
        print(f"   Tokens: {learner.tokens_learned:,}")

        print("\n🎯 Sample Generations:")
        for prompt in ["Hello", "The sky", "What"]:
            output = learner.generate(prompt, max_length=20)
            print(f"   '{prompt}' → '{output}'")


def chat_mode(checkpoint_name='final'):
    """Chat with trained model"""
    print("=" * 70)
    print("💬 CHAT MODE")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        learner = FlashDiffusionLearner.load_checkpoint(checkpoint_name, device=device)
    except FileNotFoundError:
        print(f"❌ Checkpoint '{checkpoint_name}' not found!")
        return

    print(f"\n📊 Blocks: {len(learner.model.get_all_blocks())}")
    print(f"📊 Tokens learned: {learner.tokens_learned:,}")
    print("\nType 'quit' to exit\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            response = learner.generate(user_input, max_length=40, temperature=0.8)
            print(f"AI: {response}\n")

    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--chat':
            checkpoint = sys.argv[2] if len(sys.argv) > 2 else 'final'
            chat_mode(checkpoint)
        elif sys.argv[1] == '--help':
            print("ExpandFormer v6: Flash + Diffusion + Growth")
            print("\nUsage:")
            print("  python expandformer_v6.py              # Train")
            print("  python expandformer_v6.py --chat       # Chat")
            print("  python expandformer_v6.py --chat NAME  # Chat with specific checkpoint")
        else:
            print(f"Unknown: {sys.argv[1]}")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()