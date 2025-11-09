"""
ExpandFormer v12: Forced Abstraction Through Extreme Constraint
================================================================

YOUR NEW INSIGHT:
"Start SO TINY it can't memorize, only abstract"
"Force pattern learning through extreme constraint"
"Grow capacity only after abstraction is learned"

PHILOSOPHY:
Baby brain → Learns patterns (no room for details)
Child brain → Adds details to patterns
Adult brain → Rich detail on top of solid abstractions

This is how humans actually learn!

ARCHITECTURE:
- Stage 1 (Tiny): 16→32 dims, 1 block - FORCE abstraction
- Stage 2 (Small): 32→64 dims, 2 blocks - Add capacity
- Stage 3 (Medium): 48→96 dims, 3-4 blocks - Full model
- Only grows when absolutely stuck

REQUIREMENTS:
pip install torch tiktoken numpy

USAGE:
python expandformer_v12.py              # Train from UNREASONABLY tiny
python expandformer_v12.py --chat       # Chat mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque, Counter
from pathlib import Path
import sys
import random

try:
    import tiktoken
except ImportError:
    print("ERROR: pip install tiktoken")
    sys.exit(1)


# ============================================================================
# ABSTRACTION-FORCING COMPONENTS
# ============================================================================

class AdaptiveLossWeighter:
    """Focus on learnable patterns, defer impossible ones"""

    def __init__(self, vocab_size, initial_weight=0.3):
        self.vocab_size = vocab_size
        self.token_successes = np.zeros(vocab_size)
        self.token_attempts = np.zeros(vocab_size)
        self.initial_weight = initial_weight

    def update(self, token_id, was_correct):
        if token_id < self.vocab_size:
            self.token_attempts[token_id] += 1
            if was_correct:
                self.token_successes[token_id] += 1

    def get_weight(self, token_id):
        if token_id >= self.vocab_size:
            return self.initial_weight

        attempts = self.token_attempts[token_id]
        if attempts == 0:
            return self.initial_weight

        success_rate = self.token_successes[token_id] / attempts
        return max(self.initial_weight, success_rate ** 2)


class TinyAssociationMemory(nn.Module):
    """
    Extremely compressed associations
    Only 16 dims - MUST learn patterns, not memorize
    """

    def __init__(self, vocab_size, fast_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.fast_dim = fast_dim

        self.fast_lookup = nn.Embedding(vocab_size, fast_dim)
        self.to_output = nn.Linear(fast_dim, vocab_size)

    def forward(self, x):
        fast_embed = self.fast_lookup(x[:, -1])
        fast_logits = self.to_output(fast_embed)
        return fast_logits


class ExtremelyConservativeBlock(nn.Module):
    """
    Splits VERY RARELY - only when truly stuck for long time
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

        # MUCH longer tracking window
        self.register_buffer('recent_losses', torch.zeros(200))  # Was 100
        self.loss_idx = 0
        self.register_buffer('avg_loss', torch.tensor(0.0))
        self.register_buffer('loss_variance', torch.tensor(0.0))
        self.updates_since_birth = 0

        # Track plateau (stuck for long time)
        self.register_buffer('loss_plateau_count', torch.tensor(0))
        self.register_buffer('best_loss_seen', torch.tensor(100.0))

    def forward(self, x):
        if self.is_split and len(self.child_blocks) > 0:
            outputs = []
            for child in self.child_blocks:
                outputs.append(child(x))
            return torch.stack(outputs).mean(dim=0)

        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended
        x = x + self.ff(self.norm2(x))
        return x

    def update_difficulty(self, loss_value, token_ids=None):
        """Track with very long window for stability"""
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 200
        self.avg_loss = self.recent_losses.mean()
        self.loss_variance = self.recent_losses.var()
        self.updates_since_birth += 1

        # Track plateau (not improving)
        if loss_value < self.best_loss_seen:
            self.best_loss_seen = loss_value
            self.loss_plateau_count = 0
        else:
            self.loss_plateau_count += 1

    def should_split(self, global_avg_loss, min_updates=500):
        """
        EXTREMELY conservative - only split if:
        1. Been alive VERY long (500+ updates)
        2. AND stuck in plateau (no improvement for 100+ updates)
        3. AND significantly worse than average
        """
        if self.is_split or self.updates_since_birth < min_updates:
            return False

        # Must be plateaued (stuck) for long time
        if self.loss_plateau_count < 100:
            return False

        # AND must be significantly worse
        if self.avg_loss < global_avg_loss * 1.8:  # 80% worse, not 50%
            return False

        return True

    def split(self, num_children=2):
        """Split into specialized children"""
        if self.is_split:
            return False

        print(f"  🌱 GROWTH: {self.block_id} after {self.updates_since_birth} updates")
        print(f"     Reason: Plateaued for {self.loss_plateau_count.item()} updates")

        device = next(self.parameters()).device

        for i in range(num_children):
            child = ExtremelyConservativeBlock(
                self.hidden_dim,
                self.num_heads,
                self.vocab_size,
                dropout=0.1,
                block_id=f"{self.block_id}.{i}"
            ).to(device)

            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        noise = torch.randn_like(parent_param) * 0.02
                        child_param.data.copy_(parent_param.data + noise)

            self.child_blocks.append(child)

        self.is_split = True

        for param in self.parameters():
            param.requires_grad = False
        for child in self.child_blocks:
            for param in child.parameters():
                param.requires_grad = True

        return True


# ============================================================================
# ABSTRACTION-FORCING TRANSFORMER
# ============================================================================

class AbstractionForcingTransformer(nn.Module):
    """
    YOUR VISION: Start SO TINY it must learn abstractions

    Stage 1: 16→32 dims, 1 block  (~2M params)
    Stage 2: 32→64 dims, 2 blocks (~4M params)
    Stage 3: 48→96 dims, 3+ blocks (~8M params)

    Only grows when truly plateaued
    """

    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32,
                 context_len=256, num_blocks=1, num_heads=1, dropout=0.1,
                 max_splits=2):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_splits = max_splits

        print(f"🔬 FORCING ABSTRACTION: Starting with {embed_dim}→{hidden_dim} dims")
        print(f"   This is UNREASONABLY tiny - must learn patterns, not facts!")

        # FAST PATH: Extremely compressed
        self.fast_memory = TinyAssociationMemory(vocab_size, fast_dim=16)

        # SLOW PATH: Tiny start
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, embed_dim))

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Start with just 1 block!
        self.blocks = nn.ModuleList([
            ExtremelyConservativeBlock(hidden_dim, num_heads, vocab_size, dropout, block_id=f"B{i}")
            for i in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.slow_output = nn.Linear(hidden_dim, vocab_size)

        # Two-speed mixing
        self.fast_confidence = 0.5  # Start balanced

        # Stats
        self.total_updates = 0
        self.total_splits = 0

        # Track overall progress for growth decisions
        self.register_buffer('global_best_loss', torch.tensor(100.0))
        self.register_buffer('global_plateau_count', torch.tensor(0))

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
        return pos_enc

    def forward(self, x, return_both=False):
        batch_size, seq_len = x.shape

        # FAST PATH
        fast_logits = self.fast_memory(x)

        # SLOW PATH
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

        combined = self.fast_confidence * fast_logits + (1 - self.fast_confidence) * slow_logits
        return combined

    def increase_slow_confidence(self):
        self.fast_confidence = max(0.1, self.fast_confidence * 0.998)

    def get_all_blocks(self):
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
        """Check if should split - EXTREMELY conservative"""
        if self.total_splits >= self.max_splits:
            return False

        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(global_avg_loss, min_updates=500):
                if block.split():
                    self.total_splits += 1
                    if self.total_splits >= self.max_splits:
                        print(f"  ⚠️  Reached max splits ({self.max_splits})")
                    return True

        return False


# ============================================================================
# ABSTRACTION LEARNER
# ============================================================================

class AbstractionLearner:
    """
    Learns abstractions first (tiny model)
    Then adds detail capacity (growth)
    """

    def __init__(self, model, tokenizer, lr=0.0005, device='cuda'):
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

        self.warmup_steps = 500
        self.current_step = 0

        self.loss_weighter = AdaptiveLossWeighter(model.vocab_size)
        self.recent_losses = deque(maxlen=100)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        return self.base_lr

    def update_lr(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_forced_abstraction(self, texts, sample_every=50):
        """
        Train with extreme constraint → forces abstraction
        """
        print("\n" + "=" * 70)
        print("🔬 FORCED ABSTRACTION LEARNING")
        print("=" * 70)
        print("\nPhilosophy: Start SO TINY the model MUST learn patterns")
        print("Can't memorize → Must abstract → Then grow capacity\n")
        print(f"Initial: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"This is ~5x smaller than typical!\n")

        # Simple curriculum - just increase context
        curriculum = [
            ("Abstraction Phase", 32, 4),  # Learn patterns with tiny brain
            ("Refinement Phase", 64, 3),  # Add some detail
            ("Integration Phase", 128, 2),  # Full context
        ]

        test_prompts = self._sample_prompts(texts, n=5)

        for stage_name, context_len, epochs in curriculum:
            print(f"\n{'=' * 70}")
            print(f"🧠 {stage_name.upper()} (context={context_len})")
            print(f"{'=' * 70}")
            print(f"Model size: {sum(p.numel() for p in self.model.parameters()):,} params")
            print(f"Blocks: {len(self.model.get_all_blocks())}\n")

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                total_loss = 0
                num_updates = 0
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

                        # Adaptive weighted loss
                        raw_loss = F.cross_entropy(logits[:, -1, :], y, reduction='none')
                        weight = self.loss_weighter.get_weight(y_token)
                        loss = (raw_loss * weight).mean()

                        predicted = logits[:, -1, :].argmax(dim=-1).item()
                        was_correct = (predicted == y_token)
                        self.loss_weighter.update(y_token, was_correct)

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

                        # Update block difficulties
                        for block in self.model.get_all_blocks():
                            block.update_difficulty(loss.item(), x)

                        if num_updates % 100 == 0:
                            self.model.increase_slow_confidence()

                        # Live output
                        if num_updates - last_sample >= sample_every:
                            last_sample = num_updates
                            avg_loss = np.mean(list(self.recent_losses))

                            print(f"\n{'─' * 70}")
                            print(f"📊 Update {num_updates:5d} | Loss: {avg_loss:.4f} | "
                                  f"Blocks: {len(self.model.get_all_blocks())} | "
                                  f"Splits: {self.model.total_splits}/{self.model.max_splits}")
                            print(f"{'─' * 70}")

                            for prompt in test_prompts:
                                try:
                                    output = self.generate(prompt, max_length=30, temperature=0.9)
                                    display = output if len(output) <= 60 else output[:60] + "..."
                                    print(f"  '{prompt}' → {display}")
                                except:
                                    print(f"  '{prompt}' → [error]")

                            print(f"{'─' * 70}\n")

                    if (text_idx + 1) % 20 == 0:
                        avg_loss = total_loss / num_updates if num_updates > 0 else 0
                        print(f"  Progress: {text_idx + 1}/{len(texts)} | loss={avg_loss:.4f}")

                # Check for splits after epoch
                avg_loss = np.mean(list(self.recent_losses))
                if self.model.check_and_split(avg_loss):
                    print(f"  🌱 Model grew! Now {len(self.model.get_all_blocks())} blocks")
                    print(f"     Total params: {sum(p.numel() for p in self.model.parameters()):,}")

                print(f"✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}\n")

        print("\n✅ Abstraction learning complete!")
        print(f"   Learned abstractions with tiny brain, then added capacity")

    def _sample_prompts(self, texts, n=5):
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

                for token in recent_tokens:
                    if token < logits.shape[-1]:
                        logits[0, token] /= repetition_penalty

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

            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded.strip() in ['.', '!', '?'] and random.random() < 0.3:
                    break
            except:
                pass

        self.model.train()
        return self.tokenizer.decode(generated)

    def save(self, name, save_dir='checkpoints_v12'):
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
                'max_splits': self.model.max_splits,
            },
            'stats': {
                'total_updates': self.model.total_updates,
                'total_splits': self.model.total_splits,
                'current_step': self.current_step,
                'fast_confidence': self.model.fast_confidence,
            },
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {name}")


# ============================================================================
# MAIN
# ============================================================================

def train_pipeline():
    print("=" * 70)
    print("🔬 ExpandFormer v12: Forced Abstraction Learning")
    print("=" * 70)
    print("\nYOUR INSIGHT:")
    print("  Start UNREASONABLY tiny → Force pattern learning")
    print("  Can't memorize → Must abstract")
    print("  Learn compression → Then add capacity\n")

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

    # Create UNREASONABLY TINY model
    print("🔬 Creating FORCED ABSTRACTION model...")
    model = AbstractionForcingTransformer(
        vocab_size=vocab_size,
        embed_dim=16,  # TINY!
        hidden_dim=32,  # TINY!
        context_len=256,
        num_blocks=1,  # Just ONE block!
        num_heads=1,  # Just ONE head!
        dropout=0.1,
        max_splits=2  # Only 2 splits allowed
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ {total_params:,} parameters")
    print(f"   This is ~5x SMALLER than v11!")
    print(f"   MUST learn abstractions, can't memorize\n")

    # Train
    learner = AbstractionLearner(model, tokenizer, lr=0.0005, device=device)
    learner.train_forced_abstraction(training_texts, sample_every=50)

    # Save
    learner.save("final")

    print("\n✅ Training complete!")
    print(f"\n📊 Final Stats:")
    print(f"   Blocks: {len(model.get_all_blocks())}")
    print(f"   Splits: {model.total_splits}/{model.max_splits}")
    print(f"   Final params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n💡 Learned abstractions first, then added capacity!")


def chat_mode(checkpoint='final'):
    print("=" * 70)
    print("💬 CHAT MODE")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        learner = AbstractionLearner.load(checkpoint, device=device)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    print(f"\n📊 Model Stats:")
    print(f"   Blocks: {len(learner.model.get_all_blocks())}")
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
        elif sys.argv[1] == '--help':
            print(__doc__)
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()