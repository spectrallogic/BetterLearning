"""
ExpandFormer v2: FIXED Organic Growth with Attention
=====================================================

FIXES FROM v1:
1. ✓ Fixed splitting math (actually works now!)
2. ✓ Added self-attention (for semantic clustering)
3. ✓ Increased context window (128 bytes for real chat)
4. ✓ Added interactive chat interface
5. ✓ Better generation (top-k sampling)
6. ✓ Semantic pattern tracking

CORE PHILOSOPHY:
- Starts TINY (baby brain)
- Grows when confused (novelty-triggered)
- Learns continuously (online)
- Clusters semantically (attention)
- Learns more from you than itself
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque
import threading
import os


class AttentionBlock(nn.Module):
    """
    Block with self-attention for semantic clustering
    Can split when confused about diverse patterns
    """
    def __init__(self, hidden_dim, num_heads=2, block_id="root", parent_loss=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_id = block_id

        # Self-attention for semantic understanding
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Splitting state
        self.is_split = False
        self.child_blocks = nn.ModuleList()

        # Confusion tracking - FIXED VERSION
        self.register_buffer('recent_losses', torch.zeros(20))  # More history
        self.loss_idx = 0
        self.register_buffer('avg_confusion', torch.tensor(0.0))

        # Better birth confusion initialization
        if parent_loss is not None:
            self.register_buffer('birth_confusion', parent_loss.clone())
        else:
            self.register_buffer('birth_confusion', torch.tensor(2.0))  # Lower baseline

        self.updates_since_birth = 0

        # Pattern memory for semantic clustering
        self.pattern_memory = deque(maxlen=100)  # Remember recent patterns

    def forward(self, x, return_attention=False):
        """
        Forward with self-attention
        x: [batch, seq, hidden]
        """
        if self.is_split and len(self.child_blocks) > 0:
            # Split: combine children
            # Each child processes same input, we average outputs
            outputs = []
            for child in self.child_blocks:
                out = child(x, return_attention=False)
                outputs.append(out)

            # Average child outputs (ensemble-like)
            return torch.stack(outputs).mean(dim=0)
        else:
            # Self-attention
            attended, attn_weights = self.attention(x, x, x)
            x = self.norm1(x + attended)

            # Feed-forward
            x = self.norm2(x + self.ff(x))

            if return_attention:
                return x, attn_weights
            return x

    def update_confusion(self, loss_value, input_embedding=None):
        """
        Track confusion - FIXED to actually trigger splits
        """
        if self.is_split:
            return

        # Track loss directly (not multiplied by gradients!)
        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 20
        self.avg_confusion = self.recent_losses.mean()
        self.updates_since_birth += 1

        # Track pattern diversity
        if input_embedding is not None:
            self.pattern_memory.append(input_embedding.detach().cpu())

    def compute_pattern_diversity(self):
        """
        Measure how diverse the patterns this block sees
        High diversity = needs to split
        """
        if len(self.pattern_memory) < 10:
            return 0.0

        patterns = torch.stack(list(self.pattern_memory)[-50:])  # Last 50

        # Compute pairwise distances
        mean_pattern = patterns.mean(dim=0)
        distances = torch.norm(patterns - mean_pattern, dim=1)

        return distances.mean().item()

    def should_split(self, loss_threshold=1.5, min_updates=30):
        """
        FIXED: Actually works now!
        Split if: loss is high OR pattern diversity is high
        """
        if self.is_split:
            return False

        if self.updates_since_birth < min_updates:
            return False

        # Criterion 1: High loss (confusion)
        loss_confused = self.avg_confusion > loss_threshold

        # Criterion 2: High pattern diversity
        diversity = self.compute_pattern_diversity()
        diversity_confused = diversity > 0.5

        return loss_confused or diversity_confused

    def split(self, num_children=2):
        """Split into specialized children"""
        if self.is_split:
            return False

        print(f"  🌱 SPLIT {self.block_id}: loss={self.avg_confusion:.3f}, diversity={self.compute_pattern_diversity():.3f}")

        device = next(self.parameters()).device

        for i in range(num_children):
            child = AttentionBlock(
                self.hidden_dim,
                self.num_heads,
                block_id=f"{self.block_id}.{i}",
                parent_loss=self.avg_confusion
            ).to(device)

            # Initialize from parent with noise (specialization)
            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        child_param.data.copy_(
                            parent_param.data + torch.randn_like(parent_param) * 0.02
                        )

            self.child_blocks.append(child)

        self.is_split = True

        # Freeze parent
        for param in self.parameters():
            param.requires_grad = False

        # But keep children trainable
        for child in self.child_blocks:
            for param in child.parameters():
                param.requires_grad = True

        return True


class ExpandFormerV2(nn.Module):
    """
    V2: Fixed organic growth with attention
    """
    def __init__(
        self,
        vocab_size=256,
        embed_dim=96,
        hidden_dim=192,
        context_len=128,  # Much larger for real conversation
        num_initial_blocks=2,
        num_heads=4,
        split_threshold=1.5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.split_threshold = split_threshold

        print("=" * 70)
        print("🌱 ExpandFormer V2 - Fixed Organic Growth + Attention")
        print("=" * 70)
        print(f"  Starting: {num_initial_blocks} blocks, {hidden_dim}D, {num_heads} heads")
        print(f"  Context: {context_len} bytes (can remember conversations!)")
        print(f"  Split threshold: {split_threshold} loss")
        print()

        # Byte embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(context_len, embed_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Attention blocks (start small!)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, block_id=f"B{i}")
            for i in range(num_initial_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Stats
        self.total_updates = 0
        self.total_splits = 0

        self._print_stats()

    def _create_pos_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding"""
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc

    def _print_stats(self):
        """Show model size"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_blocks = len(self.get_all_blocks())

        print(f"  📊 Blocks: {num_blocks} | Params: {trainable:,} ({trainable/1e3:.1f}K)")

    def forward(self, x):
        """
        Forward pass
        x: [batch, seq_len] byte values
        """
        batch_size, seq_len = x.shape

        # Embed + position
        h = self.embedding(x)
        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        # Project
        h = self.input_proj(h)

        # Through attention blocks
        for block in self.blocks:
            h = block(h)

        # Output
        h = self.output_norm(h)
        logits = self.output(h)

        return logits

    def get_all_blocks(self):
        """Get all leaf blocks"""
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
        """Check all blocks and split confused ones"""
        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(loss_threshold=self.split_threshold):
                if block.split():
                    self.total_splits += 1
                    self._print_stats()
                    return True

        return False


class SmartLearner:
    """
    Online learner with better generation and tracking
    """
    def __init__(self, model, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

        self.losses = []
        self.split_history = []

        print(f"🧠 Smart Learner ready (lr={lr})")
        print()

    def learn_sequence(self, text, source_weight=1.0, verbose=False):
        """
        Learn from text
        source_weight: 1.0=user, 0.3=self
        """
        if isinstance(text, str):
            bytes_data = [ord(c) for c in text if ord(c) < 256]
        else:
            bytes_data = text

        if len(bytes_data) < 2:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        # Learn byte by byte
        for i in range(len(bytes_data) - 1):
            # Build context
            context_start = max(0, i - self.model.context_len + 1)
            context = bytes_data[context_start:i+1]

            # Pad
            while len(context) < self.model.context_len:
                context = [0] + context

            x = torch.tensor([context[-self.model.context_len:]], dtype=torch.long, device=self.device)
            y = torch.tensor([bytes_data[i+1]], dtype=torch.long, device=self.device)

            # Train
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)

            weighted_loss = loss * source_weight
            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update block confusions with ACTUAL loss (not gradient!)
            blocks = self.model.get_all_blocks()
            for block in blocks:
                # Get input embedding for pattern tracking
                with torch.no_grad():
                    h = self.model.embedding(x)
                    h = self.model.input_proj(h)
                    input_emb = h[:, -1, :].mean(dim=0)  # Last position

                block.update_confusion(loss.item(), input_emb)

            total_loss += loss.item()
            num_updates += 1
            self.model.total_updates += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        self.losses.append(avg_loss)

        # Check for splits every 25 updates
        if self.model.total_updates % 25 == 0:
            if self.model.check_and_split():
                self.split_history.append(self.model.total_updates)
                # Recreate optimizer with new parameters
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    weight_decay=0.01,
                    betas=(0.9, 0.95)
                )

        return avg_loss

    def generate(self, prompt, max_length=100, temperature=0.8, top_k=40):
        """
        Generate with top-k sampling (better quality)
        """
        self.model.eval()

        if isinstance(prompt, str):
            result = prompt
            bytes_data = [ord(c) for c in prompt if ord(c) < 256]
        else:
            result = ""
            bytes_data = list(prompt)

        for _ in range(max_length):
            # Context
            context = bytes_data[-self.model.context_len:]
            while len(context) < self.model.context_len:
                context = [0] + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()

            # Stop conditions
            if next_byte == 0:  # Null byte
                break
            if next_byte == ord('.') and np.random.random() < 0.4:  # Sometimes stop at period
                try:
                    result += chr(next_byte)
                except:
                    pass
                break

            try:
                char = chr(next_byte)
                # Stop at newline sometimes
                if char == '\n' and np.random.random() < 0.5:
                    break
                result += char
                bytes_data.append(next_byte)
            except:
                break

        self.model.train()
        return result


def interactive_chat(learner, save_dir="chat_logs"):
    """
    INTERACTIVE CHAT MODE
    Learn from conversation in real-time!
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("💬 INTERACTIVE CHAT MODE")
    print("=" * 70)
    print("Commands:")
    print("  'quit' - Exit chat")
    print("  'stats' - Show model statistics")
    print("  'save' - Save current model")
    print()
    print("The AI learns from YOU as you chat!")
    print("It values YOUR words more than its own.")
    print("=" * 70)
    print()

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\n👋 Goodbye! The AI remembers everything we talked about.")
                break

            if user_input.lower() == 'stats':
                blocks = learner.model.get_all_blocks()
                print(f"\n📊 Model Stats:")
                print(f"   Blocks: {len(blocks)}")
                print(f"   Splits: {learner.model.total_splits}")
                print(f"   Updates: {learner.model.total_updates}")
                print(f"   Avg Loss: {np.mean(learner.losses[-10:]) if learner.losses else 0:.3f}")
                print()
                continue

            if user_input.lower() == 'save':
                path = os.path.join(save_dir, f"model_{int(time.time())}.pt")
                torch.save(learner.model.state_dict(), path)
                print(f"💾 Saved to {path}\n")
                continue

            # Learn from user (high weight!)
            loss = learner.learn_sequence(user_input, source_weight=1.0)

            # Generate response
            response = learner.generate(
                user_input,
                max_length=80,
                temperature=0.85,
                top_k=40
            )

            # Clean up response (remove prompt)
            if response.startswith(user_input):
                response = response[len(user_input):].strip()

            print(f"AI: {response}")

            # Learn from own response (low weight)
            learner.learn_sequence(response, source_weight=0.3)

            # Save to history
            conversation_history.append({
                'user': user_input,
                'ai': response,
                'loss': loss,
                'blocks': len(learner.model.get_all_blocks())
            })

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def demo_fixed_splitting():
    """
    Test that splitting ACTUALLY works now
    """
    print("=" * 70)
    print("TEST: Fixed Splitting Mechanism")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ExpandFormerV2(
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        context_len=32,
        num_initial_blocks=2,
        num_heads=2,
        split_threshold=1.2  # Lower threshold
    )

    learner = SmartLearner(model, lr=0.01, device=device)

    print("Phase 1: Simple pattern...")
    for i in range(15):
        loss = learner.learn_sequence("ABABABABA")
        if i % 5 == 0:
            print(f"  Update {i}: Loss={loss:.3f}, Blocks={len(model.get_all_blocks())}")

    print(f"\n✓ After simple: {len(model.get_all_blocks())} blocks\n")

    print("Phase 2: Complex pattern (should trigger split)...")
    for i in range(30):
        loss = learner.learn_sequence("ABAbfgshfABBABA XYZ123")
        if i % 5 == 0:
            print(f"  Update {i}: Loss={loss:.3f}, Blocks={len(model.get_all_blocks())}")

    print(f"\n✓ After complex: {len(model.get_all_blocks())} blocks")
    print(f"✓ Total splits: {model.total_splits}")
    print()

    if model.total_splits > 0:
        print("✅ SUCCESS! Splitting works!")
    else:
        print("⚠️  No splits occurred")

    print("=" * 70)


def demo_text_learning():
    """
    Demo learning real text
    """
    print("\n" * 2)
    print("=" * 70)
    print("TEST: Real Text Learning")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ExpandFormerV2(
        vocab_size=256,
        embed_dim=96,
        hidden_dim=192,
        context_len=64,
        num_initial_blocks=2,
        num_heads=4,
        split_threshold=1.5
    )

    learner = SmartLearner(model, lr=0.003, device=device)

    texts = [
        "Hello, how are you today?",
        "I am learning to understand language.",
        "Neural networks are fascinating.",
        "The cat sat on the mat.",
        "Machine learning helps computers learn.",
    ]

    print("Training on sample sentences...\n")

    for epoch in range(8):
        epoch_loss = 0
        for text in texts:
            loss = learner.learn_sequence(text, source_weight=1.0)
            epoch_loss += loss

        avg_loss = epoch_loss / len(texts)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.3f}, Blocks={len(model.get_all_blocks())}")

        if epoch % 2 == 1:
            prompt = "Hello"
            gen = learner.generate(prompt, max_length=40, temperature=0.7)
            print(f"  Gen: '{gen}'")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test 1: Verify splitting works
    demo_fixed_splitting()

    # Test 2: Real text learning
    demo_text_learning()

    # Test 3: Interactive chat (uncomment to use)
    print("\n" * 2)
    print("=" * 70)
    print("Ready for interactive chat!")
    print("Uncomment the line below in main() to start chatting")
    print("=" * 70)

    # Uncomment this to start chat:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = ExpandFormerV2(context_len=128, num_heads=4)
    # learner = SmartLearner(model, lr=0.005, device=device)
    # interactive_chat(learner)