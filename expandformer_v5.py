"""
ExpandFormer v5: Attention-Based Hierarchical Memory
=====================================================

FEATURES:
✓ Standard GPT-style tokenization (tiktoken)
✓ Memory hierarchy emerges from attention patterns
✓ Per-token learning rates (abstract = slow, episodic = fast)
✓ Real-time continuous learning
✓ Auto-save with periodic checkpoints
✓ Chat mode with any saved model

REQUIREMENTS:
pip install torch tiktoken numpy

USAGE:
python expandformer_v5.py            # Train
python expandformer_v5.py --chat     # Chat with saved model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict
import threading
from pathlib import Path
from datetime import datetime
import sys
import json

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed!")
    print("Install with: pip install tiktoken")
    sys.exit(1)


class AttentionCentralityTracker:
    """Tracks which tokens are 'central' in attention patterns"""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # How often each token is attended TO (in-degree)
        self.attention_received = torch.zeros(vocab_size)

        # How often each token attends to OTHERS (out-degree)
        self.attention_given = torch.zeros(vocab_size)

        # Total observations per token
        self.token_counts = torch.zeros(vocab_size)

        # Computed centrality scores
        self.centrality_scores = torch.zeros(vocab_size)

        # Update counter
        self.update_count = 0

    def update(self, token_ids, attention_weights):
        """
        Update centrality based on attention patterns

        Args:
            token_ids: [seq_len] - the tokens
            attention_weights: [num_heads, seq_len, seq_len] - attention matrix
        """
        # Average across heads
        avg_attention = attention_weights.mean(dim=0)  # [seq_len, seq_len]
        seq_len = token_ids.shape[0]

        for i in range(seq_len):
            token_id = token_ids[i].item()

            if token_id >= self.vocab_size:
                continue

            # In-degree: How much attention does this token RECEIVE?
            received = avg_attention[:, i].sum().item()
            self.attention_received[token_id] += received

            # Out-degree: How much attention does this token GIVE?
            given = avg_attention[i, :].sum().item()
            self.attention_given[token_id] += given

            self.token_counts[token_id] += 1

        self.update_count += 1

    def compute_centrality(self):
        """Compute abstraction scores from attention patterns"""
        mask = self.token_counts > 0

        normalized_received = torch.zeros_like(self.attention_received)
        normalized_given = torch.zeros_like(self.attention_given)

        normalized_received[mask] = (
            self.attention_received[mask] / self.token_counts[mask]
        )
        normalized_given[mask] = (
            self.attention_given[mask] / self.token_counts[mask]
        )

        # Centrality = weighted combination
        # Being attended TO is more important (abstract concepts)
        self.centrality_scores = (
            0.7 * normalized_received +
            0.3 * normalized_given
        )

        # Normalize to [0, 1]
        max_score = self.centrality_scores.max()
        if max_score > 0:
            self.centrality_scores = self.centrality_scores / max_score

        return self.centrality_scores

    def get_memory_depth(self, token_id):
        """Get memory depth classification"""
        score = self.centrality_scores[token_id].item()

        if score > 0.7:
            return 'abstract'
        elif score > 0.3:
            return 'contextual'
        else:
            return 'episodic'


class MemoryAwareAttention(nn.Module):
    """Self-attention that tracks its own patterns"""

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

        # Track attention centrality
        self.centrality_tracker = AttentionCentralityTracker(vocab_size)

        # Store last attention for analysis
        self.last_attention_weights = None

    def forward(self, x, token_ids=None):
        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Store for tracking
        self.last_attention_weights = attn_weights.detach()

        # Update centrality if training and we have token IDs
        if self.training and token_ids is not None:
            # Only track for first item in batch
            self.centrality_tracker.update(
                token_ids[0],
                attn_weights[0].detach().cpu()
            )

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

        # Standard embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Per-token learning rate multipliers
        self.register_buffer('lr_multipliers', torch.ones(vocab_size))

    def forward(self, token_ids):
        return self.embedding(token_ids)

    def update_lr_multipliers(self, centrality_scores):
        """
        Update LR multipliers based on attention centrality
        High centrality → Low LR (abstract, hard to forget)
        Low centrality → High LR (episodic, easy to forget)
        """
        # Invert centrality: abstract learns slowly
        self.lr_multipliers = 1.0 - 0.9 * centrality_scores

        # Clamp to [0.05, 1.0]
        self.lr_multipliers = torch.clamp(self.lr_multipliers, 0.05, 1.0)


class AttentionBlock(nn.Module):
    """Attention block with memory tracking"""

    def __init__(self, hidden_dim, num_heads, vocab_size, block_id="B0"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_id = block_id

        self.attention = MemoryAwareAttention(hidden_dim, num_heads, vocab_size)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Loss tracking for potential splitting
        self.register_buffer('recent_losses', torch.zeros(20))
        self.loss_idx = 0
        self.register_buffer('avg_confusion', torch.tensor(0.0))
        self.updates_since_birth = 0

    def forward(self, x, token_ids=None):
        attended = self.attention(x, token_ids)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        return x

    def update_confusion(self, loss_value):
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 20
        self.avg_confusion = self.recent_losses.mean()
        self.updates_since_birth += 1


class ExpandFormerV5(nn.Module):
    """Attention-based hierarchical memory transformer"""

    def __init__(self, vocab_size, embed_dim=96, hidden_dim=192,
                 context_len=128, num_blocks=2, num_heads=4):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # Hierarchical embedding
        self.embedding = HierarchicalEmbedding(vocab_size, embed_dim)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(context_len, embed_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, vocab_size, block_id=f"B{i}")
            for i in range(num_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Stats
        self.total_updates = 0
        self.tokens_learned = 0

        # Memory hierarchy stats
        self.last_hierarchy_update = 0

    def _create_pos_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x, track_attention=False):
        batch_size, seq_len = x.shape

        # Get token IDs for attention tracking
        token_ids = x if track_attention else None

        # Embed
        h = self.embedding(x)
        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        h = self.input_proj(h)

        # Through attention blocks (with tracking)
        for block in self.blocks:
            h = block(h, token_ids)

        h = self.output_norm(h)
        logits = self.output(h)

        return logits

    def get_centrality_stats(self, tokenizer):
        """Get current memory hierarchy statistics"""
        # Aggregate centrality from all blocks
        all_centrality = torch.zeros(self.vocab_size)
        num_trackers = 0

        for block in self.blocks:
            centrality = block.attention.centrality_tracker.compute_centrality()
            all_centrality += centrality
            num_trackers += 1

        if num_trackers > 0:
            all_centrality /= num_trackers

        # Get top abstract and episodic tokens
        top_k = 10

        # Most abstract (highest centrality)
        top_abstract_scores, top_abstract_ids = torch.topk(all_centrality, top_k)

        # Most episodic (lowest centrality, but only if observed)
        mask = all_centrality > 0
        if mask.sum() > 0:
            valid_centrality = all_centrality.clone()
            valid_centrality[~mask] = float('inf')
            bottom_episodic_scores, bottom_episodic_ids = torch.topk(
                valid_centrality, min(top_k, mask.sum()), largest=False
            )
        else:
            bottom_episodic_scores = torch.tensor([])
            bottom_episodic_ids = torch.tensor([])

        return {
            'all_centrality': all_centrality,
            'abstract_tokens': [(tokenizer.decode([idx.item()]), score.item())
                               for idx, score in zip(top_abstract_ids, top_abstract_scores)],
            'episodic_tokens': [(tokenizer.decode([idx.item()]), score.item())
                               for idx, score in zip(bottom_episodic_ids, bottom_episodic_scores)],
            'mean_centrality': all_centrality[all_centrality > 0].mean().item() if (all_centrality > 0).sum() > 0 else 0,
        }

    def update_memory_hierarchy(self):
        """Update per-token learning rates based on attention patterns"""
        # Aggregate centrality scores from all blocks
        all_centrality = torch.zeros(self.vocab_size)
        num_trackers = 0

        for block in self.blocks:
            centrality = block.attention.centrality_tracker.compute_centrality()
            all_centrality += centrality
            num_trackers += 1

        if num_trackers > 0:
            all_centrality /= num_trackers

        # Update embedding layer LR multipliers
        self.embedding.update_lr_multipliers(all_centrality)

        self.last_hierarchy_update = self.total_updates

    def save_checkpoint(self, path):
        """Save model"""
        checkpoint = {
            'model_state': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'context_len': self.context_len,
                'num_blocks': len(self.blocks),
            },
            'stats': {
                'total_updates': self.total_updates,
                'tokens_learned': self.tokens_learned,
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'])
        self.total_updates = checkpoint['stats']['total_updates']
        self.tokens_learned = checkpoint['stats']['tokens_learned']


class MemoryAwareOptimizer:
    """Optimizer with per-token learning rate scaling"""

    def __init__(self, model, base_lr=0.003):
        self.model = model
        self.base_lr = base_lr

        # Standard optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # Scale embedding gradients by per-token LR multipliers
        with torch.no_grad():
            embedding_weight = self.model.embedding.embedding.weight

            if embedding_weight.grad is not None:
                lr_multipliers = self.model.embedding.lr_multipliers

                # Scale each token's gradient
                for token_id in range(self.model.vocab_size):
                    if token_id < embedding_weight.grad.shape[0]:
                        embedding_weight.grad[token_id] *= lr_multipliers[token_id]

        # Standard step
        self.optimizer.step()

    def get_param_group_lr(self, idx=0):
        return self.optimizer.param_groups[idx]['lr']


class HierarchicalLearner:
    """Real-time learner with attention-based memory"""

    def __init__(self, model, tokenizer, lr=0.003, device='cuda', save_dir='checkpoints_v5'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.optimizer = MemoryAwareOptimizer(model, base_lr=lr)

        self.recent_loss_window = deque(maxlen=50)
        self.running = True
        self.tokens_learned = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes

        # For auto-generation
        self.generation_prompts = ["Hello", "How", "What", "The"]

    def learn_text(self, text, show_stats=False):
        """Learn from text with hierarchical memory"""
        token_ids = self.tokenizer.encode(text)

        if len(token_ids) < 2:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        for i in range(len(token_ids) - 1):
            # Get context
            context_start = max(0, i - self.model.context_len + 1)
            context = token_ids[context_start:i+1]

            # Pad if needed
            pad_token = 0  # Use 0 as pad for tiktoken
            while len(context) < self.model.context_len:
                context = [pad_token] + context

            x = torch.tensor([context[-self.model.context_len:]], dtype=torch.long, device=self.device)
            y = torch.tensor([token_ids[i+1]], dtype=torch.long, device=self.device)

            # Train
            self.optimizer.zero_grad()

            logits = self.model(x, track_attention=True)
            loss = F.cross_entropy(logits[:, -1, :], y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update block confusion
            for block in self.model.blocks:
                block.update_confusion(loss.item())

            total_loss += loss.item()
            num_updates += 1
            self.model.total_updates += 1
            self.tokens_learned += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        self.recent_loss_window.append(avg_loss)

        # Update memory hierarchy every 100 steps
        if self.model.total_updates % 100 == 0:
            self.model.update_memory_hierarchy()
            if show_stats:
                self.print_hierarchy_stats()

        # Auto-save every N minutes
        if time.time() - self.last_save_time > self.save_interval:
            self.save_checkpoint()
            self.last_save_time = time.time()

        return avg_loss

    def generate(self, prompt, max_length=30, temperature=0.8):
        """Generate text"""
        self.model.eval()

        token_ids = self.tokenizer.encode(prompt)

        for _ in range(max_length):
            context = token_ids[-self.model.context_len:]
            pad_token = 0
            while len(context) < self.model.context_len:
                context = [pad_token] + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            token_ids.append(next_token)

            # Stop at end of sentence sometimes
            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded in ['.', '!', '?', '\n'] and np.random.random() < 0.3:
                    break
            except:
                pass

        self.model.train()

        return self.tokenizer.decode(token_ids)

    def get_stats(self):
        """Get training statistics"""
        avg_loss = np.mean(list(self.recent_loss_window)) if self.recent_loss_window else 0
        runtime = time.time() - self.start_time

        return {
            'loss': avg_loss,
            'updates': self.model.total_updates,
            'tokens': self.tokens_learned,
            'runtime': runtime,
            'tokens_per_sec': self.tokens_learned / runtime if runtime > 0 else 0,
        }

    def print_hierarchy_stats(self):
        """Print memory hierarchy statistics"""
        stats = self.model.get_centrality_stats(self.tokenizer)

        print(f"\n🧠 Memory Hierarchy (mean centrality: {stats['mean_centrality']:.3f})")

        print(f"\n  📌 Abstract (slow learning, hard to forget):")
        for token, score in stats['abstract_tokens'][:5]:
            lr_mult = self.model.embedding.lr_multipliers[self.tokenizer.encode(token)[0]].item()
            print(f"     '{token}': centrality={score:.3f}, LR={lr_mult:.3f}x")

        print(f"\n  💭 Episodic (fast learning, easy to forget):")
        for token, score in stats['episodic_tokens'][:5]:
            try:
                token_id = self.tokenizer.encode(token)[0]
                lr_mult = self.model.embedding.lr_multipliers[token_id].item()
                print(f"     '{token}': centrality={score:.3f}, LR={lr_mult:.3f}x")
            except:
                pass

    def save_checkpoint(self, name=None):
        """Save checkpoint"""
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"checkpoint_{timestamp}"

        # Save model
        model_path = self.save_dir / f"{name}_model.pt"
        self.model.save_checkpoint(model_path)

        # Save learner state
        learner_state = {
            'tokens_learned': self.tokens_learned,
            'start_time': self.start_time,
            'lr': self.optimizer.get_param_group_lr(),
        }
        learner_path = self.save_dir / f"{name}_learner.json"
        with open(learner_path, 'w') as f:
            json.dump(learner_state, f, indent=2)

        print(f"\n💾 Checkpoint saved: {name}")

    @classmethod
    def load_checkpoint(cls, checkpoint_name, device='cuda'):
        """Load checkpoint"""
        save_dir = Path('checkpoints_v5')

        # Load model
        model_path = save_dir / f"{checkpoint_name}_model.pt"
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']

        # Create tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create model
        model = ExpandFormerV5(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            context_len=config['context_len'],
            num_blocks=config['num_blocks'],
        )
        model.load_checkpoint(model_path)

        # Load learner state
        learner_path = save_dir / f"{checkpoint_name}_learner.json"
        with open(learner_path, 'r') as f:
            learner_state = json.load(f)

        # Create learner
        learner = cls(model, tokenizer, lr=learner_state['lr'], device=device)
        learner.tokens_learned = learner_state['tokens_learned']
        learner.start_time = learner_state['start_time']

        print(f"✅ Checkpoint loaded: {checkpoint_name}")
        return learner

    def auto_generate_loop(self):
        """Background thread for auto-generation"""
        while self.running:
            time.sleep(5)
            if self.tokens_learned > 0:
                prompt = np.random.choice(self.generation_prompts)
                output = self.generate(prompt, max_length=20, temperature=0.7)
                stats = self.get_stats()
                print(f"\n💭 [updates={stats['updates']}, loss={stats['loss']:.3f}] '{prompt}' → '{output}'")
                sys.stdout.flush()


def train_demo():
    """Main training demo"""
    print("=" * 70)
    print("🚀 ExpandFormer v5: Attention-Based Hierarchical Memory")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

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
                    # Chunk by paragraphs or fixed size
                    for i in range(0, len(lines), 6):
                        chunk = '\n'.join(lines[i:i+6])
                        if chunk:
                            training_texts.append(chunk)
                print(f"   ✓ {file_path.name}: {len(training_texts)} chunks")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

    if not training_texts:
        print("⚠️  No training data found, using sample...")
        training_texts = [
            "Hello, how are you? I am doing well.",
            "The sky is blue. The grass is green.",
            "What is your name? My name is Claude.",
        ]

    print(f"\n✓ {len(training_texts)} training samples")
    print()

    # Create tokenizer (GPT-2 style)
    print("🔤 Loading tokenizer (GPT-2)...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"   ✓ Vocab size: {vocab_size:,}")
    print()

    # Create model
    print("🧠 Creating model...")
    model = ExpandFormerV5(
        vocab_size=vocab_size,
        embed_dim=96,
        hidden_dim=192,
        context_len=128,
        num_blocks=3,
        num_heads=4,
    )
    print(f"   ✓ {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Create learner
    learner = HierarchicalLearner(model, tokenizer, lr=0.003, device=device)

    # Start auto-generation
    print("🎯 Starting auto-generation (every 5 seconds)...")
    gen_thread = threading.Thread(target=learner.auto_generate_loop, daemon=True)
    gen_thread.start()
    print()

    # Training loop
    print("=" * 70)
    print("📚 REAL-TIME TRAINING WITH HIERARCHICAL MEMORY")
    print("=" * 70)
    print()
    print("Watch attention patterns create memory hierarchy!")
    print("Press Ctrl+C to stop and save")
    print("-" * 70)
    print()

    try:
        sample_idx = 0
        while sample_idx < len(training_texts):
            text = training_texts[sample_idx]

            # Learn
            loss = learner.learn_text(text, show_stats=False)

            # Progress
            if sample_idx % 10 == 0:
                stats = learner.get_stats()
                print(f"📖 Sample {sample_idx}/{len(training_texts)} | "
                      f"Updates: {stats['updates']} | "
                      f"Loss: {stats['loss']:.3f} | "
                      f"Tokens: {stats['tokens']:,}")

                # Show hierarchy every 50 samples
                if sample_idx % 50 == 0 and sample_idx > 0:
                    learner.print_hierarchy_stats()

            sample_idx += 1
            time.sleep(0.05)

        print()
        print("✓ Training data processed!")
        print()
        print("Interactive mode - type to continue training (Ctrl+C to exit):")
        print()

        # Interactive
        while True:
            user_input = input("You: ").strip()
            if user_input:
                loss = learner.learn_text(user_input, show_stats=False)
                stats = learner.get_stats()
                print(f"   [Learned! Loss: {loss:.3f}]")

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
        learner.running = False
        time.sleep(1)

        # Save
        print("\n💾 Saving final model...")
        learner.save_checkpoint("final")
        learner.print_hierarchy_stats()

        # Stats
        stats = learner.get_stats()
        print()
        print("=" * 70)
        print("📊 FINAL STATS")
        print("=" * 70)
        print(f"Updates: {stats['updates']:,}")
        print(f"Tokens: {stats['tokens']:,}")
        print(f"Loss: {stats['loss']:.3f}")
        print(f"Runtime: {stats['runtime']/60:.1f} min")
        print()

        # Generations
        print("🎯 Final Generations:")
        for prompt in ["Hello", "The sky", "What is"]:
            output = learner.generate(prompt, max_length=25)
            print(f"   '{prompt}' → '{output}'")

        print()
        print("✓ Use 'python expandformer_v5.py --chat' to chat!")


def chat_mode(checkpoint_name='final'):
    """Chat with saved model"""
    print("=" * 70)
    print("💬 CHAT MODE")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        learner = HierarchicalLearner.load_checkpoint(checkpoint_name, device=device)
    except FileNotFoundError:
        print(f"❌ Checkpoint '{checkpoint_name}' not found!")
        return

    stats = learner.get_stats()
    print()
    print(f"📊 Model: {stats['updates']:,} updates, {stats['tokens']:,} tokens learned")
    print()
    print("Commands: 'stats', 'hierarchy', 'quit'")
    print("=" * 70)
    print()

    temperature = 0.8
    max_length = 50

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.lower() == 'stats':
                stats = learner.get_stats()
                print(f"\n📊 Stats:")
                print(f"   Updates: {stats['updates']:,}")
                print(f"   Tokens: {stats['tokens']:,}")
                print(f"   Loss: {stats['loss']:.3f}\n")
                continue

            if user_input.lower() == 'hierarchy':
                learner.print_hierarchy_stats()
                continue

            if user_input.lower() == 'settings':
                print(f"\nCurrent: temp={temperature}, max_len={max_length}")
                try:
                    t = input("Temperature (0.1-2.0) [enter to skip]: ").strip()
                    if t:
                        temperature = max(0.1, min(2.0, float(t)))
                    l = input("Max length (10-200) [enter to skip]: ").strip()
                    if l:
                        max_length = max(10, min(200, int(l)))
                    print(f"✓ temp={temperature}, max_len={max_length}\n")
                except:
                    print("❌ Invalid\n")
                continue

            if not user_input:
                continue

            response = learner.generate(user_input, max_length=max_length, temperature=temperature)
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
            print("ExpandFormer v5: Attention-Based Hierarchical Memory")
            print()
            print("Usage:")
            print("  python expandformer_v5.py              # Train")
            print("  python expandformer_v5.py --chat       # Chat with 'final'")
            print("  python expandformer_v5.py --chat NAME  # Chat with checkpoint")
        else:
            print(f"Unknown: {sys.argv[1]}")
    else:
        train_demo()


if __name__ == "__main__":
    main()