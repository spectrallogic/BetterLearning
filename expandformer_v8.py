"""
ExpandFormer v7: Natural Learning Architecture
===============================================

PHILOSOPHY: Learn Like Humans
- No tricks, no keyword forcing, no hardcoded behaviors
- Pure architectural solutions
- Learns from context and patterns naturally
- Flexible for ANY input data
- Real-time growth based on confusion, not predetermined rules

KEY PRINCIPLES:
✓ No centrality-based learning rate manipulation
✓ No aggressive focus weighting
✓ Natural curriculum: simple → complex
✓ Proper regularization prevents mode collapse
✓ Repetition penalties for coherent generation
✓ Test prompts sampled from actual training data
✓ Architecture learns structure, not hardcoded

ARCHITECTURE IMPROVEMENTS:
✓ Relative positional encoding
✓ Dropout for generalization
✓ Layer normalization before attention (Pre-LN)
✓ Better token mixing in feedforward
✓ Gradient clipping and warmup
✓ Dynamic context windows
✓ Repetition-aware generation

REQUIREMENTS:
pip install torch tiktoken numpy scikit-learn

USAGE:
python expandformer_v7.py              # Train on any data
python expandformer_v7.py --chat       # Chat mode
python expandformer_v7.py --fast       # Fast mode (less verbose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict, Counter
from pathlib import Path
import sys
import json
import random

try:
    import tiktoken
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install tiktoken scikit-learn")
    sys.exit(1)


# ============================================================================
# CORPUS ANALYZER - No hardcoding, pure statistics
# ============================================================================

class CorpusAnalyzer:
    """
    Analyzes corpus to understand its structure
    BUT doesn't manipulate learning - just provides info for smart initialization
    """

    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.token_freq = Counter()
        self.token_bigrams = Counter()
        self.corpus_stats = {}

    def analyze(self, texts):
        """Analyze corpus - pure statistics, no manipulation"""
        print("\n" + "=" * 70)
        print("📊 ANALYZING CORPUS")
        print("=" * 70)

        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)

            # Track frequency
            for tok in tokens:
                self.token_freq[tok] += 1

            # Track bigrams for context
            for i in range(len(tokens) - 1):
                self.token_bigrams[(tokens[i], tokens[i + 1])] += 1

        print(f"✓ Analyzed {len(all_tokens):,} tokens")
        print(f"✓ Unique tokens: {len(self.token_freq)}")
        print(f"✓ Unique bigrams: {len(self.token_bigrams)}")

        # Compute statistics
        self.corpus_stats = {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(self.token_freq),
            'unique_bigrams': len(self.token_bigrams),
            'avg_token_freq': np.mean(list(self.token_freq.values())),
        }

        # Sample some common phrases (for test prompts later)
        common_starts = []
        for (t1, t2), count in self.token_bigrams.most_common(100):
            try:
                bigram_text = self.tokenizer.decode([t1, t2])
                if len(bigram_text.strip()) > 2:  # Not just punctuation
                    common_starts.append(bigram_text.strip())
            except:
                pass

        self.corpus_stats['common_phrases'] = common_starts[:20]

        print(f"\n📈 Corpus Statistics:")
        print(f"   Average token frequency: {self.corpus_stats['avg_token_freq']:.1f}")
        print(f"   Vocabulary coverage: {len(self.token_freq) / self.vocab_size * 100:.1f}%")

        return self.corpus_stats


# ============================================================================
# IMPROVED ATTENTION - Relative positions, dropout, better mixing
# ============================================================================

class NaturalAttention(nn.Module):
    """
    Attention that learns naturally without manipulation
    - Relative positional encoding
    - Dropout for regularization
    - No artificial centrality tracking
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, max_rel_pos=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Relative positional embeddings (learn distance relationships)
        self.max_rel_pos = max_rel_pos
        self.rel_pos_embed = nn.Embedding(2 * max_rel_pos + 1, self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def _get_relative_positions(self, seq_len):
        """Compute relative position indices"""
        positions = torch.arange(seq_len, device=self.rel_pos_embed.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Clip to max range
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_rel_pos,
            self.max_rel_pos
        )

        # Shift to positive indices
        relative_positions = relative_positions + self.max_rel_pos

        return relative_positions

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative positional bias
        rel_pos = self._get_relative_positions(seq_len)
        rel_pos_embed = self.rel_pos_embed(rel_pos)  # [seq_len, seq_len, head_dim]

        # Compute relative position scores
        rel_scores = torch.einsum('bhqd,qkd->bhqk', Q, rel_pos_embed)
        scores = scores + rel_scores / math.sqrt(self.head_dim)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output


class AdaptiveBlock(nn.Module):
    """
    Transformer block that can split when confused
    No semantic specialization tricks - just learns naturally
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, block_id="B0"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_id = block_id

        # Pre-LN architecture (more stable)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = NaturalAttention(hidden_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Split mechanism
        self.is_split = False
        self.child_blocks = nn.ModuleList()

        # Performance tracking
        self.register_buffer('recent_losses', torch.zeros(50))  # Longer window
        self.loss_idx = 0
        self.register_buffer('avg_loss', torch.tensor(0.0))
        self.register_buffer('loss_variance', torch.tensor(0.0))
        self.updates_since_birth = 0

    def forward(self, x):
        if self.is_split and len(self.child_blocks) > 0:
            outputs = []
            for child in self.child_blocks:
                out = child(x)
                outputs.append(out)
            return torch.stack(outputs).mean(dim=0)

        # Pre-LN: normalize before attention
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def update_performance(self, loss_value):
        """Track performance with variance"""
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 50
        self.avg_loss = self.recent_losses.mean()
        self.loss_variance = self.recent_losses.var()
        self.updates_since_birth += 1

    def should_split(self, global_avg_loss, min_updates=100):
        """
        Split if:
        1. Trained enough
        2. Loss is high compared to average
        3. Loss is volatile (high variance)
        """
        if self.is_split or self.updates_since_birth < min_updates:
            return False

        # High loss (struggling)
        if self.avg_loss > global_avg_loss * 1.3:
            return True

        # High variance (inconsistent)
        if self.loss_variance > 0.5:
            return True

        return False

    def split(self, num_children=2):
        """Split into children"""
        if self.is_split:
            return False

        print(f"  🌱 Splitting {self.block_id}: loss={self.avg_loss:.3f}, var={self.loss_variance:.3f}")

        device = next(self.parameters()).device

        for i in range(num_children):
            child = AdaptiveBlock(
                self.hidden_dim,
                self.num_heads,
                dropout=0.1,
                block_id=f"{self.block_id}.{i}"
            ).to(device)

            # Inherit weights with small noise
            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        noise = torch.randn_like(parent_param) * 0.02
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


# ============================================================================
# MAIN MODEL
# ============================================================================

class NaturalTransformer(nn.Module):
    """
    Transformer that learns naturally without tricks
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 context_len=256, num_blocks=6, num_heads=8, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # Standard embedding - no manipulation
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, embed_dim))

        # Project to hidden dimension
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaptiveBlock(hidden_dim, num_heads, dropout, block_id=f"B{i}")
            for i in range(num_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Stats
        self.total_updates = 0
        self.total_splits = 0

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def _create_pos_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding"""
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Embed
        h = self.embedding(x)
        h = self.embed_dropout(h)

        # Add positional encoding
        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        # Project
        h = self.input_proj(h)

        # Transform
        for block in self.blocks:
            h = block(h)

        # Output
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

    def check_and_split(self, global_avg_loss):
        """Check if any blocks should split"""
        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(global_avg_loss, min_updates=100):
                if block.split():
                    self.total_splits += 1
                    return True

        return False


# ============================================================================
# NATURAL LEARNER - No tricks, pure learning
# ============================================================================

class NaturalLearner:
    """
    Learns naturally through curriculum and proper training
    No centrality manipulation, no aggressive weighting
    """

    def __init__(self, model, tokenizer, corpus_stats, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.corpus_stats = corpus_stats
        self.device = device

        self.base_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )

        # Warmup scheduler
        self.warmup_steps = 1000
        self.current_step = 0

        # Loss tracking
        self.recent_losses = deque(maxlen=100)

    def get_lr(self):
        """Warmup learning rate schedule"""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        return self.base_lr

    def update_lr(self):
        """Update learning rate"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_natural_curriculum(self, texts, sample_every=50):
        """
        Natural curriculum learning:
        - Start with short contexts (easier)
        - Gradually increase context (harder)
        - Like how humans learn: simple → complex
        """
        print("\n" + "=" * 70)
        print("🎓 NATURAL CURRICULUM LEARNING")
        print("=" * 70)
        print("\nLearning progression: Short contexts → Long contexts")
        print("Like a human: Simple patterns first, then complex structure\n")

        # Curriculum stages: gradually increase context
        curriculum = [
            ("Foundation", 32, 2),  # Very short context, 2 epochs
            ("Building", 64, 2),  # Medium context, 2 epochs
            ("Mastery", 128, 2),  # Long context, 2 epochs
            ("Polish", 256, 1),  # Full context, 1 epoch
        ]

        # Sample test prompts from actual training data
        test_prompts = self._sample_test_prompts(texts, n=5)

        for stage_name, context_len, epochs in curriculum:
            print(f"\n{'=' * 70}")
            print(f"📚 STAGE: {stage_name} (context={context_len})")
            print(f"{'=' * 70}\n")

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                total_loss = 0
                num_updates = 0
                last_sample = 0

                # Process texts
                for text_idx, text in enumerate(texts):
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) < 2:
                        continue

                    # Train on this text
                    for i in range(1, len(tokens)):
                        # Get context (limited by current curriculum stage)
                        context_start = max(0, i - context_len)
                        context = tokens[context_start:i]

                        # Pad if needed
                        if len(context) < context_len:
                            context = [0] * (context_len - len(context)) + context

                        x = torch.tensor([context[-context_len:]], dtype=torch.long, device=self.device)
                        y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

                        # Forward
                        self.optimizer.zero_grad()
                        logits = self.model(x)
                        loss = F.cross_entropy(logits[:, -1, :], y)

                        # Backward
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.update_lr()
                        self.optimizer.step()

                        # Track
                        total_loss += loss.item()
                        num_updates += 1
                        self.current_step += 1
                        self.model.total_updates += 1
                        self.recent_losses.append(loss.item())

                        # Update blocks
                        for block in self.model.get_all_blocks():
                            block.update_performance(loss.item())

                        # Live output
                        if num_updates - last_sample >= sample_every:
                            last_sample = num_updates
                            avg_loss = np.mean(list(self.recent_losses))

                            print(f"\n{'─' * 70}")
                            print(
                                f"📊 Update {num_updates:5d} | Loss: {avg_loss:.4f} | LR: {self.get_lr():.6f} | Blocks: {len(self.model.get_all_blocks())}")
                            print(f"{'─' * 70}")

                            for prompt in test_prompts:
                                try:
                                    output = self.generate(
                                        prompt,
                                        max_length=30,
                                        temperature=0.9,
                                        repetition_penalty=1.2
                                    )
                                    display = output if len(output) <= 60 else output[:60] + "..."
                                    print(f"  '{prompt}' → {display}")
                                except Exception as e:
                                    print(f"  '{prompt}' → [error]")

                            print(f"{'─' * 70}\n")

                    # Progress
                    if (text_idx + 1) % 20 == 0:
                        avg_loss = total_loss / num_updates if num_updates > 0 else 0
                        print(f"  Progress: {text_idx + 1}/{len(texts)} texts | loss={avg_loss:.4f}")

                # Check for splits
                avg_loss = np.mean(list(self.recent_losses))
                if self.model.check_and_split(avg_loss):
                    print(f"  🌱 Model grew! Now {len(self.model.get_all_blocks())} blocks")

                print(f"✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}\n")

        print("\n✅ Natural curriculum training complete!")

    def _sample_test_prompts(self, texts, n=5):
        """Sample test prompts from actual training data"""
        prompts = []

        # Get random snippets from training data
        for _ in range(n * 3):  # Sample more, filter later
            text = random.choice(texts)
            tokens = self.tokenizer.encode(text)

            if len(tokens) > 3:
                # Random starting position
                start = random.randint(0, len(tokens) - 3)
                # Take 1-3 tokens
                length = random.randint(1, min(3, len(tokens) - start))
                prompt_tokens = tokens[start:start + length]

                try:
                    prompt = self.tokenizer.decode(prompt_tokens).strip()
                    if len(prompt) > 0 and len(prompt) < 30:
                        prompts.append(prompt)
                except:
                    pass

        # Return diverse set
        unique_prompts = list(set(prompts))
        return unique_prompts[:n] if len(unique_prompts) >= n else unique_prompts

    def generate(self, prompt, max_length=50, temperature=0.9, repetition_penalty=1.2, top_p=0.9):
        """
        Natural generation with repetition penalty and nucleus sampling
        No tricks - just good generation practices
        """
        self.model.eval()

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)

        # Track recent tokens for repetition penalty
        recent_tokens = deque(maxlen=20)

        for _ in range(max_length):
            context = generated[-self.model.context_len:]
            if len(context) < self.model.context_len:
                context = [0] * (self.model.context_len - len(context)) + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature

                # Apply repetition penalty
                for token in recent_tokens:
                    if token < logits.shape[-1]:
                        logits[0, token] /= repetition_penalty

                # Nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            recent_tokens.append(next_token)

            # Stop on natural endings
            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded.strip() in ['.', '!', '?'] and random.random() < 0.3:
                    break
            except:
                pass

        self.model.train()
        return self.tokenizer.decode(generated)

    def learn_realtime(self, text):
        """Real-time learning"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        for i in range(1, len(tokens)):
            context_start = max(0, i - self.model.context_len)
            context = tokens[context_start:i]

            if len(context) < self.model.context_len:
                context = [0] * (self.model.context_len - len(context)) + context

            x = torch.tensor([context[-self.model.context_len:]], dtype=torch.long, device=self.device)
            y = torch.tensor([tokens[i]], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.update_lr()
            self.optimizer.step()

            total_loss += loss.item()
            num_updates += 1
            self.current_step += 1
            self.model.total_updates += 1
            self.recent_losses.append(loss.item())

        return total_loss / num_updates if num_updates > 0 else 0.0

    def save(self, name, save_dir='checkpoints_v7'):
        """Save model"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'hidden_dim': self.model.hidden_dim,
                'context_len': self.model.context_len,
            },
            'stats': {
                'total_updates': self.model.total_updates,
                'total_splits': self.model.total_splits,
                'current_step': self.current_step,
            }
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {name}")

    @classmethod
    def load(cls, name, device='cuda', save_dir='checkpoints_v7'):
        """Load model"""
        checkpoint = torch.load(Path(save_dir) / f"{name}.pt", map_location='cpu')
        config = checkpoint['config']

        tokenizer = tiktoken.get_encoding("gpt2")

        model = NaturalTransformer(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            context_len=config['context_len'],
        )

        model.load_state_dict(checkpoint['model_state'])

        learner = cls(model, tokenizer, {}, device=device)
        learner.optimizer.load_state_dict(checkpoint['optimizer_state'])
        learner.current_step = checkpoint['stats']['current_step']
        learner.model.total_updates = checkpoint['stats']['total_updates']
        learner.model.total_splits = checkpoint['stats']['total_splits']

        print(f"✅ Loaded: {name}")
        return learner


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def train_pipeline(fast_mode=False):
    """Natural training pipeline - no tricks"""
    print("=" * 70)
    print("🚀 ExpandFormer v7: Natural Learning Architecture")
    print("=" * 70)
    print("\nPHILOSOPHY: Pure architecture, no tricks")
    print("Learn from patterns like humans do\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load ANY training data
    training_dir = Path("training_data")
    training_texts = []

    if training_dir.exists():
        print("📂 Loading training data...")
        for file_path in training_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    # Split into reasonable chunks
                    lines = [line.strip() for line in text.split('\n') if line.strip()]

                    # Adaptive chunking based on content
                    chunk_size = 10 if len(lines) > 1000 else 6

                    for i in range(0, len(lines), chunk_size):
                        chunk = '\n'.join(lines[i:i + chunk_size])
                        if len(chunk) > 10:  # Meaningful content
                            training_texts.append(chunk)

                print(f"   ✓ {file_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {file_path.name}: {e}")

    if not training_texts:
        print("⚠️  No training data found, using demo samples...")
        training_texts = [
            "The sun rises in the east and sets in the west.",
            "I enjoy reading books about science and history.",
            "What is your favorite color? Mine is blue.",
            "Learning new things is always exciting and rewarding.",
        ]

    print(f"\n✓ Loaded {len(training_texts)} text chunks\n")

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab size: {vocab_size:,}\n")

    # Analyze corpus (no manipulation, just stats)
    analyzer = CorpusAnalyzer(tokenizer, vocab_size)
    corpus_stats = analyzer.analyze(training_texts)

    # Create model
    print(f"\n🧠 Creating model...")
    model = NaturalTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        context_len=128,
        num_blocks=6,
        num_heads=8,
        dropout=0.1
    )
    print(f"✓ {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create learner
    learner = NaturalLearner(
        model, tokenizer, corpus_stats,
        lr=0.0003, device=device
    )

    # Natural curriculum training
    sample_freq = 100 if fast_mode else 50
    learner.train_natural_curriculum(training_texts, sample_every=sample_freq)

    # Save
    learner.save("final")

    # Interactive learning
    print("\n" + "=" * 70)
    print("💬 INTERACTIVE LEARNING")
    print("=" * 70)
    print("\nType messages to continue learning (Ctrl+C to stop)\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ['quit', 'exit']:
                break

            # Learn from input
            loss = learner.learn_realtime(user_input)

            # Generate response
            response = learner.generate(user_input, max_length=40, temperature=0.9, repetition_penalty=1.2)

            print(f"AI: {response}")
            print(f"   [Loss: {loss:.3f}, Blocks: {len(model.get_all_blocks())}]\n")

    except KeyboardInterrupt:
        pass

    print("\n✅ Training complete!")
    learner.save("final_interactive")

    print(f"\n📊 Final Stats:")
    print(f"   Total blocks: {len(model.get_all_blocks())}")
    print(f"   Total splits: {model.total_splits}")
    print(f"   Total updates: {model.total_updates:,}")


def chat_mode(checkpoint='final'):
    """Chat with trained model"""
    print("=" * 70)
    print("💬 CHAT MODE")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        learner = NaturalLearner.load(checkpoint, device=device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    print(f"\n📊 Model Stats:")
    print(f"   Blocks: {len(learner.model.get_all_blocks())}")
    print(f"   Updates: {learner.model.total_updates:,}")
    print("\nType 'quit' to exit\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            response = learner.generate(
                user_input,
                max_length=50,
                temperature=0.9,
                repetition_penalty=1.2
            )
            print(f"AI: {response}\n")

    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--chat':
            checkpoint = sys.argv[2] if len(sys.argv) > 2 else 'final'
            chat_mode(checkpoint)
        elif sys.argv[1] == '--fast':
            train_pipeline(fast_mode=True)
        elif sys.argv[1] == '--help':
            print("ExpandFormer v7: Natural Learning Architecture")
            print("\nUsage:")
            print("  python expandformer_v7.py           # Train (normal)")
            print("  python expandformer_v7.py --fast    # Train (less verbose)")
            print("  python expandformer_v7.py --chat    # Chat mode")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()