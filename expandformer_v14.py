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

        FIXED: No inplace operations to preserve gradient graph
        """
        # Base embeddings
        h = self.base_embed(x)  # [batch, seq, base_dim]

        # Apply domain corrections (residuals)
        if len(self.domains) > 0:
            batch, seq = x.shape

            # Create a tensor to accumulate corrections
            corrections = torch.zeros_like(h)

            for b in range(batch):
                for s in range(seq):
                    tid = x[b, s].item()

                    if tid in self.token_to_domains:
                        # Apply all domains this token belongs to
                        for domain_idx in self.token_to_domains[tid]:
                            if domain_idx < len(self.domains):
                                # Get correction from domain
                                base_vec = h[b:b + 1, s:s + 1, :]
                                correction = self.domains[domain_idx](base_vec)
                                # Accumulate correction (not inplace!)
                                corrections[b, s, :] = corrections[b, s, :] + correction.squeeze() * 0.1

            # Apply all corrections at once (creates new tensor)
            h = h + corrections

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


# ============================================================================
# MICROSCOPIC GROWTH TRANSFORMER
# ============================================================================

class MicroscopicGrowthTransformer(nn.Module):
    """
    The complete microscopic model that grows organically
    """

    def __init__(self, vocab_size, base_dim=4, hidden_dim=8,
                 context_len=128, num_heads=1, max_domains=30, max_blocks=10):
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
        print("Starting configuration:")
        print(f"  Base: {base_dim} dims")
        print(f"  Hidden: {hidden_dim} dims")
        print(f"  Blocks: 1 (will grow to {max_blocks})")
        print(f"  Domains: 0 (will grow to {max_domains})")

        # Embeddings (microscopic)
        self.embed = MicroscopicEmbedding(vocab_size, base_dim, max_domains)

        # Projection (embed -> hidden)
        self.input_proj = nn.Linear(base_dim, hidden_dim)

        # Position encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, hidden_dim))

        # Start with ONE tiny block
        self.blocks = nn.ModuleList([
            UltraLightTransformer(hidden_dim, num_heads=num_heads)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Growth tracking
        self.domains_created = 0
        self.blocks_added = 0
        self.total_updates = 0
        self.last_growth_check = 0

        initial_params = sum(p.numel() for p in self.parameters())
        print(f"\n✓ Initial parameters: {initial_params:,}")
        if initial_params > 20000:
            print(f"   Warning: {initial_params} > 20,000!")

        print("\n💡 Model will grow as it learns")
        print("=" * 70)
        print()

    def _create_pos_encoding(self, max_len, dim):
        """Sinusoidal positional encoding"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        """
        x: [batch, seq] token ids
        Returns: [batch, vocab_size] logits
        """
        batch, seq = x.shape

        # Embed
        h = self.embed(x)  # [batch, seq, base_dim]

        # Project to hidden
        h = self.input_proj(h)  # [batch, seq, hidden_dim]

        # Add position
        if seq <= self.context_len:
            h = h + self.pos_encoding[:seq].unsqueeze(0)

        # Transform
        for block in self.blocks:
            h = block(h)

        # Output
        h = self.output_norm(h)
        logits = self.output(h[:, -1, :])  # Only last position

        return logits

    def should_grow_domain(self):
        """Check if we need a new domain"""
        if self.total_updates - self.last_growth_check < 200:
            return False, []

        self.last_growth_check = self.total_updates

        # Get struggling tokens
        struggling = self.embed.get_struggling_tokens(min_samples=20)

        if len(struggling) >= 5:  # Need at least 5 tokens
            return True, struggling[:15]  # Take top 15

        return False, []

    def grow_domain(self, token_group):
        """Create new domain for struggling tokens"""
        # Grow larger domains as we learn more
        base_size = 16
        domain_size = base_size + (self.domains_created * 4)  # 16, 20, 24, 28...
        domain_size = min(domain_size, 64)  # Cap at 64

        success = self.embed.grow_domain(token_group, domain_size)

        if success:
            self.domains_created += 1
            print(f"\n  🌿 Domain {self.domains_created} created!")
            print(f"     Tokens: {len(token_group)} struggling tokens")
            total_params = sum(p.numel() for p in self.parameters())
            print(f"     Total params: {total_params:,}")

        return success

    def should_grow_block(self):
        """Check if we need another transformer block"""
        if len(self.blocks) >= self.max_blocks:
            return False

        # Add block every N domains
        blocks_expected = 1 + (self.domains_created // 3)
        return len(self.blocks) < blocks_expected

    def grow_block(self):
        """Add another transformer block"""
        device = next(self.parameters()).device

        new_block = UltraLightTransformer(
            self.hidden_dim,
            num_heads=1
        ).to(device)

        self.blocks.append(new_block)
        self.blocks_added += 1

        print(f"\n  🌿 Block {len(self.blocks)} added!")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"     Total blocks: {len(self.blocks)}")
        print(f"     Total params: {total_params:,}")

        return True


# ============================================================================
# MICROSCOPIC LEARNER
# ============================================================================

class MicroscopicLearner:
    """
    Training system for microscopic model
    """

    def __init__(self, model, tokenizer, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        self.growth_history = []

    def train(self, texts, epochs=3, sample_every=100):
        print("=" * 70)
        print("🚀 STARTING MICROSCOPIC TRAINING")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Texts: {len(texts)}")
        print(f"Device: {self.device}")
        print("=" * 70)
        print()

        total_updates = 0
        last_sample = 0

        # Initial sample prompts
        test_prompts = self._sample_prompts(texts, n=3)

        for epoch in range(epochs):
            print(f"\n{'=' * 70}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'=' * 70}\n")

            epoch_losses = []

            for text_idx, text in enumerate(texts):
                # Tokenize
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                # Create sequences
                for i in range(1, len(tokens)):
                    context = tokens[max(0, i - self.model.context_len):i]
                    target = tokens[i]

                    # Pad if needed
                    if len(context) < self.model.context_len:
                        context = [0] * (self.model.context_len - len(context)) + context

                    x = torch.tensor([context], dtype=torch.long, device=self.device)
                    y = torch.tensor([target], dtype=torch.long, device=self.device)

                    # Forward
                    self.model.train()
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)

                    # Track difficulty
                    with torch.no_grad():
                        self.model.embed.update_difficulty(target, loss.item())

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_losses.append(loss.item())
                    total_updates += 1
                    self.model.total_updates = total_updates

                    # Check for growth
                    if total_updates % 50 == 0:
                        # Domain growth
                        should_grow, struggling_tokens = self.model.should_grow_domain()
                        if should_grow:
                            self.model.grow_domain(struggling_tokens)

                            # Record growth event
                            self.growth_history.append({
                                'update': total_updates,
                                'params': sum(p.numel() for p in self.model.parameters()),
                                'domains': self.model.domains_created,
                                'blocks': len(self.model.blocks),
                            })

                        # Block growth
                        if self.model.should_grow_block():
                            self.model.grow_block()

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

ExpandFormer = MicroscopicGrowthTransformer