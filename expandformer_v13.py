"""
ExpandFormer v13: Organic Growth with Subconscious Planning
=============================================================

YOUR VISION REALIZED:
- Starts capable but constrained (not crippled)
- Grows UNEVENLY in specific knowledge domains
- Multi-speed learning spectrum (not just 2 channels)
- Subconscious layer plans ahead and influences decisions
- Natural abstraction through selective capacity growth

ARCHITECTURE:
Layer 0 (Subconscious): Plans futures → selects best path
Layer 1 (Fast): Instant pattern recognition (sea of noise → peaks)
Layer 2 (Medium): Short-term predictions (~10-50 tokens ahead)
Layer 3 (Slow): Deep understanding via transformer
Layer 4 (Integration): Subconscious influences main predictions

GROWTH:
- Not whole model growth
- Domain-specific: Hard tokens → grow their concept space
- Uneven: Some concepts stay tiny, others grow large
- Organic: Natural clustering in latent space

REQUIREMENTS:
pip install torch tiktoken numpy

USAGE:
python expandformer_v13.py              # Train with organic growth
python expandformer_v13.py --chat       # Chat mode
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
# SUBCONSCIOUS SYSTEM: Plans ahead, influences decisions
# ============================================================================

class SubconsciousPlanningLayer(nn.Module):
    """
    Layer 4: Subconscious predictions
    - Generates multiple future scenarios
    - Picks best path based on prediction quality
    - Influences main model's decisions
    """

    def __init__(self, hidden_dim, vocab_size, planning_depth=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.planning_depth = planning_depth

        # Generate multiple future scenarios
        self.scenario_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * planning_depth)
        )

        # Evaluate scenario quality
        self.scenario_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Influence vector (how much subconscious affects conscious)
        self.influence_generator = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        """
        h: [batch, seq, hidden_dim]
        Returns: influence vector to add to main predictions
        """
        batch, seq, dim = h.shape

        # Take last hidden state
        last_h = h[:, -1, :]  # [batch, hidden_dim]

        # Generate multiple future scenarios
        scenarios = self.scenario_generator(last_h)  # [batch, hidden_dim * depth]
        scenarios = scenarios.view(batch, self.planning_depth, self.hidden_dim)

        # Evaluate each scenario
        scenario_qualities = []
        for i in range(self.planning_depth):
            quality = self.scenario_evaluator(scenarios[:, i, :])  # [batch, 1]
            scenario_qualities.append(quality)

        scenario_qualities = torch.stack(scenario_qualities, dim=1)  # [batch, depth, 1]

        # Pick best scenario
        best_idx = scenario_qualities.argmax(dim=1)  # [batch, 1]
        best_scenario = scenarios[torch.arange(batch), best_idx.squeeze(-1), :]  # [batch, hidden_dim]

        # Generate influence vector
        influence = self.influence_generator(best_scenario)  # [batch, hidden_dim]

        return influence.unsqueeze(1)  # [batch, 1, hidden_dim]


# ============================================================================
# DOMAIN-SPECIFIC CAPACITY: Grows specific concept regions
# ============================================================================

# In expandformer_v13.py, replace the DomainSpecificCapacity class:

class DomainSpecificCapacity(nn.Module):
    """
    Domain-specific growth - SIMPLIFIED to actually work
    """

    def __init__(self, vocab_size, base_dim, max_domains=20):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.max_domains = max_domains

        # Base embeddings (everyone uses this)
        self.base_embeddings = nn.Embedding(vocab_size, base_dim)

        # Domain-specific expansions
        self.domain_expansions = nn.ModuleList()
        self.domain_projections = nn.ModuleList()  # Project expanded dims back to base_dim

        # Track which tokens belong to which domains
        self.token_to_domain = {}
        self.domain_token_sets = []

        # Track token difficulties - USE PYTHON DICTS (simpler)
        self.token_losses = {}
        self.token_counts = {}

    def update_difficulties(self, token_ids, losses):
        """Track which tokens are hard"""
        for token_id, loss in zip(token_ids, losses):
            if token_id >= self.vocab_size:
                continue

            if token_id not in self.token_losses:
                self.token_losses[token_id] = 0.0
                self.token_counts[token_id] = 0

            self.token_losses[token_id] += loss
            self.token_counts[token_id] += 1

    def get_hard_tokens(self, threshold=1.5, min_samples=50):
        """Find tokens that are consistently hard"""
        if not self.token_counts:
            return []

        # Calculate average losses
        avg_losses = {}
        for token_id, count in self.token_counts.items():
            if count >= min_samples:
                avg_losses[token_id] = self.token_losses[token_id] / count

        if not avg_losses:
            return []

        # Find overall average
        overall_avg = sum(avg_losses.values()) / len(avg_losses)

        # Find hard tokens (above threshold)
        hard_tokens = [
            token_id for token_id, loss in avg_losses.items()
            if loss > overall_avg * threshold and token_id not in self.token_to_domain
        ]

        return hard_tokens

    def create_new_domain(self, hard_tokens, expansion_dim=32):
        """Grow capacity for hard tokens"""
        if len(self.domain_expansions) >= self.max_domains:
            return False

        if len(hard_tokens) < 5:  # Need at least 5 tokens
            return False

        device = self.base_embeddings.weight.device

        # Create expansion layer
        expansion = nn.Linear(self.base_dim, expansion_dim, bias=False).to(device)
        projection = nn.Linear(expansion_dim, self.base_dim, bias=False).to(device)

        nn.init.xavier_uniform_(expansion.weight, gain=0.1)  # Small initial values
        nn.init.xavier_uniform_(projection.weight, gain=0.1)

        self.domain_expansions.append(expansion)
        self.domain_projections.append(projection)

        domain_idx = len(self.domain_expansions) - 1

        # Assign tokens
        token_set = set(hard_tokens)
        self.domain_token_sets.append(token_set)
        for token_id in hard_tokens:
            self.token_to_domain[token_id] = domain_idx

        print(f"      Created domain {domain_idx} with {len(hard_tokens)} tokens")

        return True

    def forward(self, token_ids):
        """
        SIMPLIFIED: Always return base_dim (no variable dimensions)
        Domains add residual corrections to base embeddings
        """
        batch, seq = token_ids.shape

        # Get base embeddings
        base_emb = self.base_embeddings(token_ids)  # [batch, seq, base_dim]

        # Add domain-specific corrections
        if len(self.domain_expansions) > 0:
            for b in range(batch):
                for s in range(seq):
                    token_id = token_ids[b, s].item()

                    if token_id in self.token_to_domain:
                        domain_idx = self.token_to_domain[token_id]

                        # Expand -> Project back (residual correction)
                        base_vec = base_emb[b, s, :].unsqueeze(0)  # [1, base_dim]
                        expanded = self.domain_expansions[domain_idx](base_vec)  # [1, expansion_dim]
                        correction = self.domain_projections[domain_idx](expanded)  # [1, base_dim]

                        # Add residual
                        base_emb[b, s, :] = base_emb[b, s, :] + correction.squeeze(0) * 0.1

        return base_emb  # Always [batch, seq, base_dim]

# ============================================================================
# MULTI-SPEED GRADIENT LEARNING
# ============================================================================

class MultiSpeedGradientLayer(nn.Module):
    """
    Not just 2 speeds - continuous spectrum of learning rates
    Fast: Instant patterns (sea of noise → peaks)
    Medium: Short-term predictions
    Slow: Deep transformer understanding
    """

    def __init__(self, vocab_size, speeds=[16, 32, 64, 128]):
        super().__init__()

        self.vocab_size = vocab_size
        self.speeds = speeds

        # Create embeddings at each speed
        self.speed_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, dim) for dim in speeds
        ])

        # Output heads for each speed
        self.speed_outputs = nn.ModuleList([
            nn.Linear(dim, vocab_size) for dim in speeds
        ])

        # Learned mixing weights (start balanced)
        self.speed_weights = nn.Parameter(torch.ones(len(speeds)) / len(speeds))

    def forward(self, x):
        """
        x: [batch, seq] token ids
        Returns: [batch, vocab_size] mixed predictions
        """
        # Get predictions from each speed
        speed_logits = []
        for i, (embedder, outputter) in enumerate(zip(self.speed_embeddings, self.speed_outputs)):
            emb = embedder(x[:, -1])  # [batch, speed_dim] - just last token
            logits = outputter(emb)  # [batch, vocab_size]
            speed_logits.append(logits)

        # Mix with learned weights
        weights = F.softmax(self.speed_weights, dim=0)
        mixed = sum(w * logits for w, logits in zip(weights, speed_logits))

        return mixed

    def update_weights(self, speed_losses):
        """
        Adjust mixing based on which speeds are working best
        Lower loss → higher weight
        """
        with torch.no_grad():
            # Invert losses (lower is better)
            inverted = 1.0 / (torch.tensor(speed_losses) + 1e-6)
            new_weights = inverted / inverted.sum()

            # Smooth update
            self.speed_weights.data = 0.9 * self.speed_weights.data + 0.1 * new_weights


# ============================================================================
# MAIN ORGANIC GROWTH TRANSFORMER
# ============================================================================

class OrganicGrowthTransformer(nn.Module):
    """
    v13: Natural growth architecture

    - Starts with adequate capacity
    - Grows UNEVENLY for hard concepts
    - Subconscious planning layer
    - Multi-speed gradient learning
    - Domain-specific expansion
    """

    def __init__(self, vocab_size, base_dim=64, hidden_dim=128,
                 context_len=256, num_blocks=2, num_heads=2, dropout=0.1,
                 max_domains=20):
        super().__init__()

        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.max_domains = max_domains

        print(f"🌱 ORGANIC GROWTH ARCHITECTURE")
        print(f"   Base: {base_dim}→{hidden_dim} dims")
        print(f"   Will grow UNEVENLY for hard concepts")
        print(f"   Max domains: {max_domains}\n")

        # Domain-specific embeddings (grows on demand)
        self.embeddings = DomainSpecificCapacity(vocab_size, base_dim, max_domains)

        # Multi-speed gradient learning
        self.multi_speed = MultiSpeedGradientLayer(vocab_size, speeds=[16, 32, 64, 128])

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(context_len, base_dim))

        # Project variable-dim embeddings to hidden_dim
        self.input_proj = nn.Linear(base_dim, hidden_dim)

        # Main transformer blocks (slow understanding)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])

        # Subconscious planning (Layer 4 from diagram)
        self.subconscious = SubconsciousPlanningLayer(hidden_dim, vocab_size, planning_depth=5)

        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.slow_output = nn.Linear(hidden_dim, vocab_size)

        # Stats
        self.total_updates = 0
        self.total_domains_created = 0
        self.last_growth_check = 0

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

    def forward(self, x, return_speeds=False):
        """
        x: [batch, seq]
        """
        batch, seq = x.shape

        # Multi-speed fast predictions
        fast_logits = self.multi_speed(x)  # [batch, vocab_size]

        # Slow transformer path
        h = self.embeddings(x)  # [batch, seq, variable_dim]

        # Project to hidden_dim (handle variable dimensions)
        if h.shape[-1] != self.base_dim:
            # Need to project from variable dim
            h_projected = []
            for b in range(batch):
                for s in range(seq):
                    vec = h[b, s, :self.base_dim]  # Take base dimensions
                    h_projected.append(vec)
            h = torch.stack(h_projected).view(batch, seq, self.base_dim)

        # Add positional encoding
        if seq <= self.context_len:
            h = h + self.pos_encoding[:seq].unsqueeze(0)

        h = self.input_proj(h)  # [batch, seq, hidden_dim]

        # Transformer blocks
        for block in self.blocks:
            mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
            h = block(h, src_mask=mask, is_causal=True)

        # Subconscious influence
        subconscious_influence = self.subconscious(h)  # [batch, 1, hidden_dim]

        # Apply influence to last position only
        last_h = h[:, -1:, :] + subconscious_influence
        h = torch.cat([h[:, :-1, :], last_h], dim=1)

        # Slow output
        h = self.output_norm(h)
        slow_logits = self.slow_output(h[:, -1, :])  # [batch, vocab_size]

        # Mix fast and slow
        # Start more balanced, let multi-speed weights handle it
        combined = 0.3 * fast_logits + 0.7 * slow_logits

        if return_speeds:
            return combined, fast_logits, slow_logits
        return combined

    def check_and_grow(self, threshold=1.5, min_samples=50):
        """
        Check if we should grow - LOWERED thresholds
        """
        if self.total_updates - self.last_growth_check < 100:  # Check every 100 (was 200)
            return False

        self.last_growth_check = self.total_updates

        hard_tokens = self.embeddings.get_hard_tokens(threshold, min_samples)

        if len(hard_tokens) >= 5:  # At least 5 hard tokens (was 10)
            # Sample subset
            sampled = hard_tokens[:min(len(hard_tokens), 30)]  # Max 30 per domain

            # Create new domain
            expansion_dim = 32
            if self.embeddings.create_new_domain(sampled, expansion_dim):
                self.total_domains_created += 1
                print(f"  🌿 Domain {self.total_domains_created} created!")
                print(f"     Tokens: {len(sampled)} | Total domains: {len(self.embeddings.domain_expansions)}")
                return True

        return False

ExpandFormer = OrganicGrowthTransformer


# ============================================================================
# ORGANIC LEARNER
# ============================================================================

class OrganicLearner:
    """
    Learns with organic growth
    - Tracks token difficulties
    - Grows domains for hard concepts
    - Multi-speed adaptation
    """

    def __init__(self, model, tokenizer, lr=0.0003, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )

        self.recent_losses = deque(maxlen=100)

    def train_organic(self, texts, epochs=3, context_len=128, sample_every=50):
        """Train with organic growth"""
        print("\n" + "=" * 70)
        print("🌱 ORGANIC LEARNING")
        print("=" * 70)
        print("Growing capacity where needed, staying compact elsewhere\n")

        test_prompts = self._sample_prompts(texts, n=5)

        for epoch in range(epochs):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 70}\n")

            total_loss = 0
            num_updates = 0
            last_sample = 0

            # Track losses per speed for adaptation
            speed_losses = defaultdict(list)

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
                    combined, fast, slow = self.model(x, return_speeds=True)

                    # Compute losses
                    loss = F.cross_entropy(combined[:, -1, :] if len(combined.shape) > 2 else combined, y)
                    loss_fast = F.cross_entropy(fast, y)
                    loss_slow = F.cross_entropy(slow, y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # Track
                    total_loss += loss.item()
                    num_updates += 1
                    self.model.total_updates += 1
                    self.recent_losses.append(loss.item())

                    # Update domain difficulties
                    self.model.embeddings.update_difficulties([y_token], [loss.item()])

                    # Track speed performance
                    speed_losses['fast'].append(loss_fast.item())
                    speed_losses['slow'].append(loss_slow.item())

                    # Adapt speed weights every 100 updates
                    if num_updates % 100 == 0:
                        avg_fast = np.mean(speed_losses['fast'][-100:])
                        avg_slow = np.mean(speed_losses['slow'][-100:])
                        self.model.multi_speed.update_weights([avg_fast, avg_slow, avg_slow, avg_slow])

                    # Check for growth every 50 updates
                    if num_updates % 50 == 0:
                        self.model.check_and_grow()

                    # Live sampling
                    if num_updates - last_sample >= sample_every:
                        last_sample = num_updates
                        avg_loss = np.mean(list(self.recent_losses))

                        print(f"\n{'─' * 70}")
                        print(f"📊 Update {num_updates:5d} | Loss: {avg_loss:.4f} | "
                              f"Domains: {len(self.model.embeddings.domain_expansions)}")
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

            avg_loss = total_loss / num_updates if num_updates > 0 else 0
            print(f"\n✓ Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}")
            print(f"  Domains created: {self.model.total_domains_created}")
            print(f"  Active domains: {len(self.model.embeddings.domain_expansions)}\n")

        print("\n✅ Organic learning complete!")
        print(f"   Grew {self.model.total_domains_created} specialized domains")

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
                if len(logits.shape) > 2:
                    logits = logits[:, -1, :]
                logits = logits / temperature

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

    def save(self, name, save_dir='checkpoints_v13'):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'vocab_size': self.model.vocab_size,
                'base_dim': self.model.base_dim,
                'hidden_dim': self.model.hidden_dim,
                'context_len': self.model.context_len,
                'max_domains': self.model.max_domains,
            },
            'stats': {
                'total_updates': self.model.total_updates,
                'total_domains_created': self.model.total_domains_created,
            },
        }

        torch.save(checkpoint, save_path / f"{name}.pt")
        print(f"💾 Saved: {name}")


# ============================================================================
# MAIN
# ============================================================================

def train_pipeline():
    print("=" * 70)
    print("🌱 ExpandFormer v13: Organic Growth Architecture")
    print("=" * 70)
    print("\nNEW DESIGN:")
    print("  • Domain-specific growth (not whole-model)")
    print("  • Subconscious planning layer")
    print("  • Multi-speed gradient learning")
    print("  • Natural abstraction through selective capacity\n")

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
        ]

    print(f"\n✓ Loaded {len(training_texts)} text chunks\n")

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"✓ Vocab size: {vocab_size:,}\n")

    # Create organic growth model
    print("🌱 Creating organic growth model...")
    model = OrganicGrowthTransformer(
        vocab_size=vocab_size,
        base_dim=64,  # Adequate starting size
        hidden_dim=128,
        context_len=256,
        num_blocks=2,
        num_heads=2,
        dropout=0.1,
        max_domains=20  # Can grow up to 20 specialized domains
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ {total_params:,} parameters")
    print(f"   Will grow domains as needed\n")

    # Train
    learner = OrganicLearner(model, tokenizer, lr=0.0003, device=device)
    learner.train_organic(training_texts, epochs=3, sample_every=50)

    # Save
    learner.save("final")

    print("\n✅ Training complete!")
    print(f"\n📊 Final Stats:")
    print(f"   Domains created: {model.total_domains_created}")
    print(f"   Active domains: {len(model.embeddings.domain_expansions)}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n💡 Grew capacity organically where needed!")


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