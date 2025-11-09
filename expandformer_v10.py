"""
ExpandFormer v9: Child-Like Learning Architecture
==================================================

YOUR VISION REALIZED:
✓ TINY START: Like a baby brain - grows only where needed
✓ TWO-SPEED LEARNING: Instant associations + slow understanding
✓ EAGER GROWTH: Multiplies when stuck (with patience)
✓ ADAPTIVE VALUE: Easy things learned fast, hard things deferred
✓ NO TRICKS: Pure architectural discovery

PHILOSOPHY:
"Start quantized, unquantize through experience"
"Learn associations instantly, understand connections slowly"
"Grow by dividing into refined parts where reality demands it"

REQUIREMENTS:
pip install torch tiktoken numpy

USAGE:
python expandformer_v9.py              # Train
python expandformer_v9.py --chat       # Chat mode
python expandformer_v9.py --tiny       # Even tinier start (32 dims!)
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
except ImportError:
    print("ERROR: tiktoken not installed!")
    print("Install with: pip install tiktoken")
    sys.exit(1)


# ============================================================================
# ADAPTIVE LOSS WEIGHTING - Value easy learning more
# ============================================================================

class AdaptiveLossWeighter:
    """
    YOUR PRINCIPLE: "Difficult to remember → values it less until repetition"

    Easy tokens get high weight = learn fast
    Hard tokens get low weight = defer until more exposure
    """

    def __init__(self, vocab_size, initial_weight=0.1):
        self.vocab_size = vocab_size
        self.token_successes = np.zeros(vocab_size)
        self.token_attempts = np.zeros(vocab_size)
        self.initial_weight = initial_weight

    def update(self, token_id, was_correct):
        """Track success/failure for each token"""
        if token_id < self.vocab_size:
            self.token_attempts[token_id] += 1
            if was_correct:
                self.token_successes[token_id] += 1

    def get_weight(self, token_id):
        """
        Get learning weight for this token

        Success rate → weight
        High success → high weight (1.0) = learn more
        Low success → low weight (0.1) = defer learning
        """
        if token_id >= self.vocab_size:
            return self.initial_weight

        attempts = self.token_attempts[token_id]
        if attempts == 0:
            return self.initial_weight

        success_rate = self.token_successes[token_id] / attempts

        # Quadratic makes the difference stark
        # 90% success → 0.81 weight
        # 50% success → 0.25 weight
        # 10% success → 0.01 weight (almost ignore)
        return max(self.initial_weight, success_rate ** 2)


# ============================================================================
# FAST ASSOCIATION MEMORY - Instant learning like flash cards
# ============================================================================

class FastAssociationMemory(nn.Module):
    """
    YOUR VISION: "Learn associations almost instantly like a child"

    This is the "flash card" layer - sees "Hello" → "Hi" and remembers
    it instantly, even if it doesn't understand WHY yet
    """

    def __init__(self, vocab_size, fast_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.fast_dim = fast_dim

        # Tiny embedding for instant associations
        self.fast_lookup = nn.Embedding(vocab_size, fast_dim)

        # Simple projection to output
        self.to_output = nn.Linear(fast_dim, vocab_size)

        # Track what we've "seen before" for confidence
        self.register_buffer('seen_bigrams', torch.zeros(vocab_size, vocab_size))

    def forward(self, x):
        """
        Fast associative lookup
        x: [batch, seq_len] tokens
        """
        # Last token lookup
        fast_embed = self.fast_lookup(x[:, -1])
        fast_logits = self.to_output(fast_embed)

        return fast_logits

    def remember_association(self, prev_token, next_token):
        """Remember this token pair was seen together"""
        if prev_token < self.vocab_size and next_token < self.vocab_size:
            self.seen_bigrams[prev_token, next_token] += 1


# ============================================================================
# DIFFICULTY-TRACKING BLOCK - Knows what it struggles with
# ============================================================================

class DifficultyAwareBlock(nn.Module):
    """
    YOUR VISION: "Grows by dividing into refined parts on areas where
    it feels that area requires more attention"

    Tracks which tokens/patterns it struggles with
    Splits to specialize when overwhelmed
    """

    def __init__(self, hidden_dim, num_heads, vocab_size, dropout=0.1, block_id="B0"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.block_id = block_id

        # Architecture
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Split tracking
        self.is_split = False
        self.child_blocks = nn.ModuleList()

        # Performance tracking
        self.register_buffer('recent_losses', torch.zeros(50))
        self.loss_idx = 0
        self.register_buffer('avg_loss', torch.tensor(0.0))
        self.register_buffer('loss_variance', torch.tensor(0.0))
        self.updates_since_birth = 0

        # CRITICAL: Track which tokens are difficult
        self.difficult_tokens = Counter()
        self.token_counts = Counter()

    def forward(self, x):
        if self.is_split and len(self.child_blocks) > 0:
            # Average children outputs
            outputs = []
            for child in self.child_blocks:
                outputs.append(child(x))
            return torch.stack(outputs).mean(dim=0)

        # Pre-LN transformer block
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended

        x = x + self.ff(self.norm2(x))
        return x

    def update_difficulty(self, loss_value, token_ids=None):
        """
        Track performance AND which tokens are hard

        This is how the block "feels" where it needs to grow
        """
        # Update rolling stats
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 50
        self.avg_loss = self.recent_losses.mean()
        self.loss_variance = self.recent_losses.var()
        self.updates_since_birth += 1

        # Track difficult tokens
        if token_ids is not None:
            for tok in token_ids.flatten().tolist():
                if tok < self.vocab_size:
                    self.difficult_tokens[tok] += loss_value
                    self.token_counts[tok] += 1

    def should_split(self, global_avg_loss, min_updates=50):
        """
        YOUR VISION: "When AI senses getting stuck it multiplies,
        but that feeling needs to take a while, not instant"

        Conditions:
        1. Been alive long enough (patience)
        2. ABSOLUTE struggle (loss > 4.0) OR relative struggle
        3. High variance (inconsistent)
        """
        if self.is_split or self.updates_since_birth < min_updates:
            return False

        # CRITICAL: Absolute threshold for early splitting
        # Even if everything struggles, blocks with loss > 4 split
        if self.avg_loss > 4.0:
            return True

        # Relative: worse than average
        if self.avg_loss > global_avg_loss * 1.2:  # Only 20% worse needed
            return True

        # High variance = inconsistent = needs specialization
        if self.loss_variance > 0.3:
            return True

        return False

    def split(self, num_children=2):
        """
        Split into specialized children

        Children inherit knowledge but specialize in difficult areas
        """
        if self.is_split:
            return False

        print(f"  🌱 Splitting {self.block_id}: loss={self.avg_loss:.3f}, var={self.loss_variance:.3f}")

        # Find difficult tokens
        if self.difficult_tokens:
            sorted_difficult = sorted(
                self.difficult_tokens.items(),
                key=lambda x: x[1] / max(self.token_counts[x[0]], 1),
                reverse=True
            )
            print(f"     Top struggles: {[tok for tok, _ in sorted_difficult[:5]]}")

        device = next(self.parameters()).device

        for i in range(num_children):
            child = DifficultyAwareBlock(
                self.hidden_dim,
                self.num_heads,
                self.vocab_size,
                dropout=0.1,
                block_id=f"{self.block_id}.{i}"
            ).to(device)

            # Inherit parent's weights with small noise
            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        noise = torch.randn_like(parent_param) * 0.02
                        child_param.data.copy_(parent_param.data + noise)

            self.child_blocks.append(child)

        self.is_split = True

        # Freeze parent, activate children
        for param in self.parameters():
            param.requires_grad = False
        for child in self.child_blocks:
            for param in child.parameters():
                param.requires_grad = True

        return True


# ============================================================================
# TWO-SPEED LEARNING MODEL - Fast associations + Slow understanding
# ============================================================================

class TwoSpeedTransformer(nn.Module):
    """
    YOUR VISION REALIZED:

    Fast path: Instant associations (flash cards)
    Slow path: Deep understanding (transformers)

    Like a child:
    - Instantly learns "Hello" → "Hi"
    - Slowly learns the nuance of greetings
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 context_len=256, num_blocks=2, num_heads=2, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # FAST PATH: Instant associations
        self.fast_memory = FastAssociationMemory(vocab_size, fast_dim=32)

        # SLOW PATH: Deep understanding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, embed_dim))

        # Project to hidden
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Difficulty-aware blocks that can split
        self.blocks = nn.ModuleList([
            DifficultyAwareBlock(hidden_dim, num_heads, vocab_size, dropout, block_id=f"B{i}")
            for i in range(num_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.slow_output = nn.Linear(hidden_dim, vocab_size)

        # Mixing: how much to trust fast vs slow (regular float, not buffer)
        self.fast_confidence = 0.3  # Start trusting fast less

        # Stats
        self.total_updates = 0
        self.total_splits = 0

        self._init_weights()

    def _init_weights(self):
        """Proper initialization"""
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

    def forward(self, x, return_both=False):
        """
        Two-speed forward pass

        Fast: Instant associations
        Slow: Deep understanding
        Combined: Best of both
        """
        batch_size, seq_len = x.shape

        # FAST PATH: Instant lookup
        fast_logits = self.fast_memory(x)

        # SLOW PATH: Deep understanding
        h = self.embedding(x)
        h = self.embed_dropout(h)

        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        slow_logits = self.slow_output(h)

        if return_both:
            return fast_logits, slow_logits

        # COMBINE: Fast for confidence, slow for understanding
        # As training progresses, we trust slow more
        combined = self.fast_confidence * fast_logits + (1 - self.fast_confidence) * slow_logits

        return combined

    def increase_slow_confidence(self):
        """As model learns, trust slow path more"""
        self.fast_confidence = max(0.1, self.fast_confidence * 0.99)

    def get_all_blocks(self):
        """Get all active blocks including split children"""

        def get_leaves(block):
            if block.is_split and len(block.child_blocks) > 0:
                leaves = []
                for child in block.child_blocks:
                    leaves.extend(get_leaves(child))
                return leaves
            return [block]

        all_leaves = []
        for block in self.blocks:
            all_leaves.extend(get_leaves(block))
        return all_leaves

    def check_and_split(self, global_avg_loss):
        """Check if any blocks should split"""
        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(global_avg_loss, min_updates=50):
                if block.split():
                    self.total_splits += 1
                    return True

        return False


# ============================================================================
# CHILD-LIKE LEARNER - Two speeds, adaptive value, eager growth
# ============================================================================

class ChildLikeLearner:
    """
    YOUR VISION: Learning like a child

    - Instant associations
    - Slow understanding
    - Values easy things
    - Grows where stuck
    """

    def __init__(self, model, tokenizer, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.base_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )

        # Warmup (shorter than v8!)
        self.warmup_steps = 500
        self.current_step = 0

        # Adaptive loss weighting
        self.loss_weighter = AdaptiveLossWeighter(model.vocab_size)

        # Loss tracking
        self.recent_losses = deque(maxlen=100)

    def get_lr(self):
        """Warmup schedule - but faster"""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        return self.base_lr

    def update_lr(self):
        """Update learning rate"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_curriculum(self, texts, sample_every=50):
        """
        Natural curriculum: short → long contexts

        Like a child: simple patterns first, complex structure later
        """
        print("\n" + "=" * 70)
        print("👶 CHILD-LIKE LEARNING CURRICULUM")
        print("=" * 70)
        print("\nStarting TINY and growing where needed...")
        print(f"Initial model: {sum(p.numel() for p in self.model.parameters()):,} parameters\n")

        # Curriculum: gradually increase context
        curriculum = [
            ("Baby Steps", 32, 3),  # Tiny context, more epochs
            ("Toddler", 64, 2),  # Small context
            ("Child", 128, 2),  # Medium context
            ("Mature", 256, 1),  # Full context
        ]

        # Sample test prompts from training data
        test_prompts = self._sample_prompts(texts, n=5)

        for stage_name, context_len, epochs in curriculum:
            print(f"\n{'=' * 70}")
            print(f"📚 {stage_name.upper()} (context={context_len})")
            print(f"{'=' * 70}\n")

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                total_loss = 0
                num_updates = 0
                last_sample = 0

                for text_idx, text in enumerate(texts):
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) < 2:
                        continue

                    # Train on this text
                    for i in range(1, len(tokens)):
                        # Get context
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

                        # Compute loss with adaptive weighting
                        raw_loss = F.cross_entropy(logits[:, -1, :], y, reduction='none')

                        # Get weight based on difficulty
                        weight = self.loss_weighter.get_weight(y_token)
                        loss = (raw_loss * weight).mean()

                        # Track success for weighting
                        predicted = logits[:, -1, :].argmax(dim=-1).item()
                        was_correct = (predicted == y_token)
                        self.loss_weighter.update(y_token, was_correct)

                        # Backward
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.update_lr()
                        self.optimizer.step()

                        # Update fast memory
                        if i > 0:
                            prev_token = tokens[i - 1]
                            self.model.fast_memory.remember_association(prev_token, y_token)

                        # Track
                        total_loss += loss.item()
                        num_updates += 1
                        self.current_step += 1
                        self.model.total_updates += 1
                        self.recent_losses.append(loss.item())

                        # Update block difficulties
                        for block in self.model.get_all_blocks():
                            block.update_difficulty(loss.item(), x)

                        # Gradually trust slow path more
                        if num_updates % 100 == 0:
                            self.model.increase_slow_confidence()

                        # Live output
                        if num_updates - last_sample >= sample_every:
                            last_sample = num_updates
                            avg_loss = np.mean(list(self.recent_losses))

                            print(f"\n{'─' * 70}")
                            print(f"📊 Update {num_updates:5d} | Loss: {avg_loss:.4f} | "
                                  f"LR: {self.get_lr():.6f} | Blocks: {len(self.model.get_all_blocks())}")
                            print(f"   Fast confidence: {self.model.fast_confidence:.3f}")
                            print(f"{'─' * 70}")

                            for prompt in test_prompts:
                                try:
                                    output = self.generate(prompt, max_length=30, temperature=0.9)
                                    display = output if len(output) <= 60 else output[:60] + "..."
                                    print(f"  '{prompt}' → {display}")
                                except Exception as e:
                                    print(f"  '{prompt}' → [error]")

                            print(f"{'─' * 70}\n")

                    # Progress
                    if (text_idx + 1) % 20 == 0:
                        avg_loss = total_loss / num_updates if num_updates > 0 else 0
                        print(f"  Progress: {text_idx + 1}/{len(texts)} | loss={avg_loss:.4f}")

                # Check for splits after each epoch
                avg_loss = np.mean(list(self.recent_losses))
                if self.model.check_and_split(avg_loss):
                    print(f"  🌱 Model grew! Now {len(self.model.get_all_blocks())} blocks")
                    print(f"     Total params: {sum(p.numel() for p in self.model.parameters()):,}")

                print(f"✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}\n")

        print("\n✅ Curriculum complete!")

    def _sample_prompts(self, texts, n=5):
        """Sample test prompts from training data"""
        prompts = []

        for _ in range(n * 3):
            text = random.choice(texts)
            tokens = self.tokenizer.encode(text)

            if len(tokens) > 3:
                start = random.randint(0, len(tokens) - 3)
                length = random.randint(1, min(3, len(tokens) - start))
                prompt_tokens = tokens[start:start + length]

                try:
                    prompt = self.tokenizer.decode(prompt_tokens).strip()
                    if 0 < len(prompt) < 30:
                        prompts.append(prompt)
                except:
                    pass

        unique = list(set(prompts))
        return unique[:n] if len(unique) >= n else unique

    def generate(self, prompt, max_length=50, temperature=0.9, repetition_penalty=1.2, top_p=0.9):
        """Generate with repetition penalty and nucleus sampling"""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        recent_tokens = deque(maxlen=20)

        for _ in range(max_length):
            context = generated[-self.model.context_len:]
            if len(context) < self.model.context_len:
                context = [0] * (self.model.context_len - len(context)) + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature

                # Repetition penalty
                for token in recent_tokens:
                    if token < logits.shape[-1]:
                        logits[0, token] /= repetition_penalty

                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            recent_tokens.append(next_token)

            # Natural endings
            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded.strip() in ['.', '!', '?'] and random.random() < 0.3:
                    break
            except:
                pass

        self.model.train()
        return self.tokenizer.decode(generated)

    def learn_realtime(self, text):
        """Real-time learning from user input"""
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
            y_token = tokens[i]
            y = torch.tensor([y_token], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)

            raw_loss = F.cross_entropy(logits[:, -1, :], y, reduction='none')
            weight = self.loss_weighter.get_weight(y_token)
            loss = (raw_loss * weight).mean()

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

    def save(self, name, save_dir='checkpoints_v9'):
        """Save checkpoint"""
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
                'fast_confidence': self.model.fast_confidence,  # Save this!
            },
            'loss_weighter': {
                'successes': self.loss_weighter.token_successes,
                'attempts': self.loss_weighter.token_attempts,
            }
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {name}")

    @classmethod
    def load(cls, name, device='cuda', save_dir='checkpoints_v9'):
        """Load checkpoint"""
        checkpoint = torch.load(Path(save_dir) / f"{name}.pt", map_location='cpu')
        config = checkpoint['config']

        tokenizer = tiktoken.get_encoding("gpt2")

        model = TwoSpeedTransformer(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            context_len=config['context_len'],
        )

        model.load_state_dict(checkpoint['model_state'])

        learner = cls(model, tokenizer, device=device)
        learner.optimizer.load_state_dict(checkpoint['optimizer_state'])
        learner.current_step = checkpoint['stats']['current_step']
        learner.model.total_updates = checkpoint['stats']['total_updates']
        learner.model.total_splits = checkpoint['stats']['total_splits']

        # Restore fast confidence (with backward compatibility)
        if 'fast_confidence' in checkpoint['stats']:
            learner.model.fast_confidence = checkpoint['stats']['fast_confidence']

        # Restore loss weighter
        learner.loss_weighter.token_successes = checkpoint['loss_weighter']['successes']
        learner.loss_weighter.token_attempts = checkpoint['loss_weighter']['attempts']

        print(f"✅ Loaded: {name}")
        return learner


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_pipeline(tiny_mode=False):
    """Train the child-like learning model"""
    print("=" * 70)
    print("👶 ExpandFormer v9: Child-Like Learning")
    print("=" * 70)
    print("\nYOUR VISION:")
    print("  • Start TINY (quantized)")
    print("  • Instant associations + Slow understanding")
    print("  • Grow where stuck")
    print("  • Value easy learning\n")

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

                    chunk_size = 10 if len(lines) > 1000 else 6

                    for i in range(0, len(lines), chunk_size):
                        chunk = '\n'.join(lines[i:i + chunk_size])
                        if len(chunk) > 10:
                            training_texts.append(chunk)

                print(f"   ✓ {file_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

    if not training_texts:
        print("⚠️  No training data, using demo samples...")
        training_texts = [
            "Hello, how are you? I am doing well.",
            "The sky is blue. The grass is green.",
            "What is your name? My name is Claude.",
            "I enjoy reading books about science.",
        ]

    print(f"\n✓ Loaded {len(training_texts)} text chunks\n")

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab size: {vocab_size:,}\n")

    # Create TINY model
    print("🧠 Creating TINY model...")
    if tiny_mode:
        embed_dim, hidden_dim, num_blocks, num_heads = 32, 64, 1, 1
        print("   ULTRA-TINY MODE: 32→64 dims, 1 block!")
    else:
        embed_dim, hidden_dim, num_blocks, num_heads = 64, 128, 2, 2
        print("   TINY MODE: 64→128 dims, 2 blocks")

    model = TwoSpeedTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_len=256,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ {total_params:,} parameters")
    print(f"   Fast memory: {sum(p.numel() for p in model.fast_memory.parameters()):,} params")
    print(f"   Slow path: {total_params - sum(p.numel() for p in model.fast_memory.parameters()):,} params\n")

    # Create learner
    learner = ChildLikeLearner(
        model, tokenizer,
        lr=0.0003, device=device
    )

    # Train with curriculum
    learner.train_curriculum(training_texts, sample_every=50)

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
            response = learner.generate(user_input, max_length=40, temperature=0.9)

            print(f"AI: {response}")
            print(f"   [Loss: {loss:.3f}, Blocks: {len(model.get_all_blocks())}, "
                  f"Fast conf: {model.fast_confidence:.3f}]\n")

    except KeyboardInterrupt:
        pass

    print("\n✅ Training complete!")
    print(f"\n📊 Final Stats:")
    print(f"   Blocks: {len(model.get_all_blocks())}")
    print(f"   Splits: {model.total_splits}")
    print(f"   Updates: {model.total_updates:,}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")


def chat_mode(checkpoint='final'):
    """Chat with trained model"""
    print("=" * 70)
    print("💬 CHAT MODE")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        learner = ChildLikeLearner.load(checkpoint, device=device)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    print(f"\n📊 Model Stats:")
    print(f"   Blocks: {len(learner.model.get_all_blocks())}")
    print(f"   Updates: {learner.model.total_updates:,}")
    print(f"   Params: {sum(p.numel() for p in learner.model.parameters()):,}")
    print("\nType 'quit' to exit\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            response = learner.generate(user_input, max_length=50, temperature=0.9)
            print(f"AI: {response}\n")

    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--chat':
            checkpoint = sys.argv[2] if len(sys.argv) > 2 else 'final'
            chat_mode(checkpoint)
        elif sys.argv[1] == '--tiny':
            train_pipeline(tiny_mode=True)
        elif sys.argv[1] == '--help':
            print("ExpandFormer v9: Child-Like Learning")
            print("\nUsage:")
            print("  python expandformer_v9.py           # Train (tiny)")
            print("  python expandformer_v9.py --tiny    # Train (ultra-tiny!)")
            print("  python expandformer_v9.py --chat    # Chat mode")
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()