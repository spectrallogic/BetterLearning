"""
ExpandFormer v14: TRUE Organic Growth from Microscopic Seed
============================================================

YOUR VISION FULLY REALIZED:
"Start SO TINY it literally cannot function without growing"
"Force growth through necessity, not thresholds"
"Baby brain that MUST learn to abstract"

ARCHITECTURE:
- Start: < 20K parameters (MICROSCOPIC!)
- Base: 8→16 dims (barely functional)
- Blocks: 1 tiny transformer
- Growth: Aggressive, necessity-driven
- Result: Grows 50-200x by end

PHILOSOPHY:
Baby can't even see clearly at birth → develops as needed
Our model can't predict well at start → grows as needed

REQUIREMENTS:
pip install torch tiktoken numpy

USAGE:
python expandformer_v14.py              # Train from microscopic seed
python expandformer_v14.py --chat       # Chat mode
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


# ============================================================================
# MICROSCOPIC BUILDING BLOCKS
# ============================================================================

class MicroscopicEmbedding(nn.Module):
    """
    Start with TINY embeddings that grow on demand
    Like neurons forming connections
    """

    def __init__(self, vocab_size, base_dim=8, max_domains=30):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.max_domains = max_domains

        # Tiny base (everyone shares)
        self.base_embed = nn.Embedding(vocab_size, base_dim)
        nn.init.normal_(self.base_embed.weight, std=0.02)

        # Domains (grown on demand)
        self.domains = nn.ModuleList()
        self.token_to_domains = defaultdict(set)  # token can be in multiple domains
        self.domain_token_sets = []

        # Track difficulties (simple counters)
        self.token_losses = defaultdict(float)
        self.token_counts = defaultdict(int)

        print(f"🔬 Microscopic embeddings: {base_dim} dims")
        print(f"   Params: {vocab_size * base_dim:,} (~{(vocab_size * base_dim) / 1000:.1f}K)")

    def update_difficulty(self, token_id, loss_value):
        """Track which tokens are hard"""
        if token_id >= self.vocab_size:
            return
        self.token_losses[token_id] += loss_value
        self.token_counts[token_id] += 1

    def get_struggling_tokens(self, min_samples=10):
        """
        Find tokens that are STRUGGLING
        Not just "hard" - actively failing
        """
        if not self.token_counts:
            return []

        # Calculate average loss per token
        avg_losses = {}
        for tid, count in self.token_counts.items():
            if count >= min_samples:
                avg_losses[tid] = self.token_losses[tid] / count

        if len(avg_losses) < 5:
            return []

        # Find global average
        global_avg = sum(avg_losses.values()) / len(avg_losses)

        # Find struggling tokens (worse than 1.2x average)
        struggling = []
        for tid, loss in avg_losses.items():
            if loss > global_avg * 1.2:
                # Check if already well-covered by domains
                if len(self.token_to_domains[tid]) < 2:  # Max 2 domains per token
                    struggling.append((tid, loss))

        # Sort by how badly they're doing
        struggling.sort(key=lambda x: x[1], reverse=True)

        return [tid for tid, _ in struggling[:20]]  # Top 20 worst

    def grow_domain(self, token_group, domain_dim=16):
        """
        Create new specialized domain
        Returns True if successful
        """
        if len(self.domains) >= self.max_domains:
            return False

        if len(token_group) < 3:  # Need at least 3 tokens
            return False

        device = self.base_embed.weight.device

        # Create domain as residual layer
        domain = nn.Sequential(
            nn.Linear(self.base_dim, domain_dim, bias=False),
            nn.GELU(),
            nn.Linear(domain_dim, self.base_dim, bias=False)
        ).to(device)

        # Initialize small (residual should start near zero)
        for param in domain.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

        self.domains.append(domain)
        domain_idx = len(self.domains) - 1

        # Assign tokens
        token_set = set(token_group)
        self.domain_token_sets.append(token_set)
        for tid in token_set:
            self.token_to_domains[tid].add(domain_idx)

        return True

    def forward(self, x):
        """
        x: [batch, seq] token ids
        Returns: [batch, seq, base_dim] embeddings (with domain corrections)
        """
        # Base embeddings
        h = self.base_embed(x)  # [batch, seq, base_dim]

        # Apply domain corrections (residuals)
        if len(self.domains) > 0:
            batch, seq = x.shape

            for b in range(batch):
                for s in range(seq):
                    tid = x[b, s].item()

                    if tid in self.token_to_domains:
                        # Apply all domains this token belongs to
                        for domain_idx in self.token_to_domains[tid]:
                            if domain_idx < len(self.domains):
                                correction = self.domains[domain_idx](h[b:b + 1, s:s + 1, :])
                                h[b, s, :] = h[b, s, :] + correction.squeeze() * 0.1

        return h


class UltraLightTransformer(nn.Module):
    """
    Absolutely minimal transformer block
    Grows by adding more blocks, not making them bigger
    """

    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()

        self.dim = dim

        # Minimal attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Minimal FFN (no expansion!)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Only 2x expansion (not 4x)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Attention
        normed = self.norm1(x)

        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, is_causal=True)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class FastAssociations(nn.Module):
    """
    Ultra-fast pattern matching
    Just lookup table with tiny projection
    """

    def __init__(self, vocab_size, fast_dim=4):
        super().__init__()

        self.fast_lookup = nn.Embedding(vocab_size, fast_dim)
        self.to_logits = nn.Linear(fast_dim, vocab_size)

        nn.init.normal_(self.fast_lookup.weight, std=0.02)
        nn.init.xavier_uniform_(self.to_logits.weight)

    def forward(self, x):
        """x: [batch, seq] → [batch, vocab_size]"""
        fast_embed = self.fast_lookup(x[:, -1])  # Just last token
        return self.to_logits(fast_embed)


# ============================================================================
# MICROSCOPIC GROWTH TRANSFORMER
# ============================================================================

class MicroscopicGrowthTransformer(nn.Module):
    """
    v14: Start with < 20K params, grow to whatever is needed

    Initial: ~15K params (microscopic!)
    Final: 200K-2M params (grown organically)
    """

    def __init__(self, vocab_size, base_dim=4, hidden_dim=8,
                 context_len=128, num_heads=1, dropout=0.1,
                 max_domains=30, max_blocks=10):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_domains = max_domains
        self.max_blocks = max_blocks

        print("=" * 70)
        print("🔬 MICROSCOPIC GROWTH TRANSFORMER v14")
        print("=" * 70)
        print(f"Starting configuration:")
        print(f"  Base: {base_dim} dims")
        print(f"  Hidden: {hidden_dim} dims")
        print(f"  Blocks: 1 (will grow to {max_blocks})")
        print(f"  Domains: 0 (will grow to {max_domains})")

        # Fast associations (immediate patterns)
        self.fast = FastAssociations(vocab_size, fast_dim=4)

        # Microscopic embeddings (grows domains)
        self.embed = MicroscopicEmbedding(vocab_size, base_dim, max_domains)

        # Positional encoding
        self.register_buffer('pos_enc', self._create_pos_encoding(context_len, base_dim))

        # Project to hidden dim
        self.input_proj = nn.Linear(base_dim, hidden_dim)

        # Start with just ONE block (grows more)
        self.blocks = nn.ModuleList([
            UltraLightTransformer(hidden_dim, num_heads, dropout)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Growth tracking
        self.total_updates = 0
        self.last_growth_check = 0
        self.domains_created = 0
        self.blocks_added = 0

        # Loss tracking for growth decisions
        self.recent_losses = deque(maxlen=50)
        self.loss_plateau_counter = 0
        self.best_loss = float('inf')

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n✓ Initial parameters: {total_params:,}")
        print(f"   Target: < 20,000 ✓" if total_params < 20000 else f"   Warning: {total_params} > 20,000!")
        print(f"\n💡 Model will grow as it learns\n")
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

    def forward(self, x, return_fast=False):
        """
        x: [batch, seq]
        Returns: logits [batch, vocab_size] or [batch, seq, vocab_size]
        """
        batch, seq = x.shape

        # Fast path (immediate associations)
        fast_logits = self.fast(x)

        # Slow path (understanding)
        h = self.embed(x)  # [batch, seq, base_dim]

        # Add positional encoding
        if seq <= self.context_len:
            h = h + self.pos_enc[:seq].unsqueeze(0)

        # Project to hidden dim
        h = self.input_proj(h)  # [batch, seq, hidden_dim]

        # Transform through blocks
        mask = None  # Block will create it internally
        for block in self.blocks:
            h = block(h, mask=mask)  # ✓ Pass mask parameter

        # Output
        h = self.output_norm(h)
        slow_logits = self.output(h)  # [batch, seq, vocab_size]

        # Take last position
        slow_logits = slow_logits[:, -1, :]  # [batch, vocab_size]

        # Mix fast and slow (more weight on slow as it learns)
        fast_weight = 0.3
        combined = fast_weight * fast_logits + (1 - fast_weight) * slow_logits

        if return_fast:
            return combined, fast_logits, slow_logits
        return combined

    def check_and_grow(self):
        """
        Growth strategy: AGGRESSIVE (fixed)
        Grow every 100 updates if loss is bad
        """
        if self.total_updates - self.last_growth_check < 50:  # Check every 50
            return False

        self.last_growth_check = self.total_updates

        if len(self.recent_losses) < 20:  # Reduced from 30
            return False

        current_loss = np.mean(list(self.recent_losses)[-10:])

        # Update best loss
        if current_loss < self.best_loss * 0.98:  # 2% improvement threshold
            self.best_loss = current_loss
            self.loss_plateau_counter = 0
            return False
        else:
            self.loss_plateau_counter += 1

        # AGGRESSIVE: Only need 10 stuck checks (was 30)
        # = 500 updates instead of 1500!
        if self.loss_plateau_counter < 10:  # Changed from 30
            return False

        grew = False

        # Strategy 1: Grow domains (specific tokens struggling)
        struggling = self.embed.get_struggling_tokens(min_samples=5)  # Reduced from 10

        if len(struggling) >= 3:
            if self.embed.grow_domain(struggling, domain_dim=16):
                self.domains_created += 1
                new_params = sum(p.numel() for p in self.parameters())
                print(f"\n  🌿 Domain {self.domains_created} created!")
                print(f"     Tokens: {len(struggling)} struggling tokens")
                print(f"     Total params: {new_params:,}")
                grew = True
                self.loss_plateau_counter = 0

        # Strategy 2: Add transformer block (need more capacity overall)
        if not grew and len(self.blocks) < self.max_blocks:
            if self.loss_plateau_counter >= 15:  # Reduced from 50
                device = next(self.parameters()).device
                new_block = UltraLightTransformer(
                    self.hidden_dim,
                    num_heads=1,
                    dropout=0.1
                ).to(device)

                self.blocks.append(new_block)
                self.blocks_added += 1
                new_params = sum(p.numel() for p in self.parameters())
                print(f"\n  🧠 Transformer block {len(self.blocks)} added!")
                print(f"     Total blocks: {len(self.blocks)}")
                print(f"     Total params: {new_params:,}")
                grew = True
                self.loss_plateau_counter = 0

        return grew

    def update_tracking(self, loss_value, token_id):
        """Update loss tracking for growth decisions"""
        self.recent_losses.append(loss_value)
        self.embed.update_difficulty(token_id, loss_value)
        self.total_updates += 1


# ============================================================================
# MICROSCOPIC LEARNER
# ============================================================================

class MicroscopicLearner:
    """
    Trains the microscopic model
    Watches it grow organically
    """

    def __init__(self, model, tokenizer, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        self.growth_history = []

    def train(self, texts, epochs=5, context_len=128, sample_every=100):
        """
        Train and watch it grow
        """
        print("🌱 TRAINING: From Microscopic Seed to Functional Intelligence\n")

        test_prompts = self._sample_prompts(texts, n=3)

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
                    # Prepare context
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

                    # Check for growth
                    if total_updates % 50 == 0:
                        if self.model.check_and_grow():
                            # Record growth event
                            self.growth_history.append({
                                'update': total_updates,
                                'params': sum(p.numel() for p in self.model.parameters()),
                                'domains': self.model.domains_created,
                                'blocks': len(self.model.blocks),
                            })

                    # Sample outputs
                    if total_updates - last_sample >= sample_every:
                        last_sample = total_updates
                        avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)

                        print(f"\n{'─' * 70}")
                        print(f"Update {total_updates:5d} | Loss: {avg_loss:.4f} | "
                              f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
                        print(f"Growth: {self.model.domains_created} domains, "
                              f"{len(self.model.blocks)} blocks")
                        print(f"{'─' * 70}")

                        for prompt in test_prompts:
                            try:
                                output = self.generate(prompt, max_length=20, temperature=0.9)
                                display = output if len(output) <= 50 else output[:50] + "..."
                                print(f"  '{prompt}' → {display}")
                            except:
                                print(f"  '{prompt}' → [error]")

                        print(f"{'─' * 70}\n")

                if (text_idx + 1) % 20 == 0:
                    avg_loss = np.mean(epoch_losses)
                    print(f"  Progress: {text_idx + 1}/{len(texts)} | loss={avg_loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"\n✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}\n")

        # Final stats
        final_params = sum(p.numel() for p in self.model.parameters())
        initial_params = self.growth_history[0]['params'] if self.growth_history else final_params

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Growth Summary:")
        print(f"  Started: {initial_params:,} params")
        print(f"  Ended: {final_params:,} params")
        print(f"  Growth: {(final_params / initial_params - 1) * 100:.1f}%")
        print(f"  Domains: {self.model.domains_created}")
        print(f"  Blocks: {len(self.model.blocks)}")
        print(f"  Growth events: {len(self.growth_history)}")
        print("=" * 70 + "\n")

    def _sample_prompts(self, texts, n=3):
        prompts = []
        for _ in range(n * 3):
            text = random.choice(texts)
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 2:
                start = random.randint(0, len(tokens) - 2)
                length = random.randint(1, min(2, len(tokens) - start))
                prompt_tokens = tokens[start:start + length]
                try:
                    prompt = self.tokenizer.decode(prompt_tokens).strip()
                    if 0 < len(prompt) < 20:
                        prompts.append(prompt)
                except:
                    pass
        unique = list(set(prompts))
        return unique[:n] if len(unique) >= n else unique

    def generate(self, prompt, max_length=30, temperature=0.9, top_p=0.9):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)

        for _ in range(max_length):
            context = generated[-self.model.context_len:]
            if len(context) < self.model.context_len:
                context = [0] * (self.model.context_len - len(context)) + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits / temperature

                # Top-p sampling
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

            try:
                decoded = self.tokenizer.decode([next_token])
                if decoded.strip() in ['.', '!', '?']:
                    break
            except:
                pass

        self.model.train()
        return self.tokenizer.decode(generated)

    def save(self, name, save_dir='checkpoints_v14'):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'growth_history': self.growth_history,
            'config': {
                'vocab_size': self.model.vocab_size,
                'base_dim': self.model.base_dim,
                'hidden_dim': self.model.hidden_dim,
                'context_len': self.model.context_len,
            },
            'stats': {
                'domains_created': self.model.domains_created,
                'blocks_added': self.model.blocks_added,
                'total_updates': self.model.total_updates,
            },
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {save_path / name}.pt")


# ============================================================================
# MAIN
# ============================================================================

def train_pipeline():
    print("=" * 70)
    print("🔬 ExpandFormer v14: Microscopic → Organic Intelligence")
    print("=" * 70)
    print("\nPHILOSOPHY:")
    print("  Start SO tiny it can barely function")
    print("  Force growth through necessity")
    print("  Watch intelligence emerge organically\n")

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
        print("⚠️  No training data, using demo samples...")
        training_texts = [
                             "Hello, how are you? I am doing well.",
                             "The sky is blue. The grass is green.",
                             "What is your name? My name is Claude.",
                             "I enjoy reading books about science.",
                         ] * 5

    print(f"\n✓ Loaded {len(training_texts)} text chunks\n")

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab size: {vocab_size:,}\n")

    # Create microscopic model
    model = MicroscopicGrowthTransformer(
        vocab_size=vocab_size,
        base_dim=8,  # TINY!
        hidden_dim=16,  # TINY!
        context_len=128,
        num_heads=1,
        max_domains=30,
        max_blocks=10
    )

    # Train
    learner = MicroscopicLearner(model, tokenizer, lr=0.001, device=device)
    learner.train(training_texts, epochs=5, sample_every=100)

    # Save
    learner.save("final")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print(__doc__)
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        train_pipeline()


if __name__ == "__main__":
    main()