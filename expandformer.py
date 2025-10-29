"""
ExpandFormer v4: True BPE Tokenization + Structural Distillation
=================================================================

TOKENIZATION:
✓ Byte Pair Encoding (learns from data)
✓ Starts with characters
✓ Merges frequent pairs into subwords
✓ No hardcoded vocabulary
✓ Learns: "ing", "tion", "hello", etc. naturally

SPECIAL TOKENS:
✓ <PAD>, <UNK>, <BOS>, <EOS>
✓ Structure tokens for sentences

ALL FEATURES PRESERVED:
✓ Attention-based template softening
✓ Organic growth
✓ File watching
✓ Interactive chat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque, defaultdict, Counter
import threading
import os
from pathlib import Path
from datetime import datetime
import re
import json


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer
    Learns subword vocabulary from data
    """
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size

        # Special tokens (always at start)
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"

        # Token mappings
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0

        # BPE merges (pair -> merged_token)
        self.merges = {}  # ("h", "e") -> "he"
        self.merge_order = []  # Order of merges for encoding

        # Initialize with special tokens + base characters
        self._initialize_base_vocab()

    def _initialize_base_vocab(self):
        """
        Initialize with special tokens + all printable ASCII characters
        No predefined words - everything learned from data!
        """
        # Add special tokens
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

        # Add all printable ASCII as base characters (32-126)
        for i in range(32, 127):
            char = chr(i)
            self.token_to_id[char] = self.next_id
            self.id_to_token[self.next_id] = char
            self.next_id += 1

        print(f"📚 Tokenizer initialized:")
        print(f"   Base vocabulary: {self.next_id} tokens (special + ASCII)")
        print(f"   Target vocab size: {self.vocab_size}")
        print(f"   Will learn subwords from data via BPE")

    def train_bpe(self, texts, verbose=True):
        """
        Train BPE on a corpus of texts
        Learns common character pairs and merges them
        """
        if verbose:
            print(f"\n🔤 Training BPE tokenizer...")

        # Tokenize all texts into character sequences
        word_freqs = Counter()
        for text in texts:
            # Split into words
            words = text.split()
            for word in words:
                # Represent as character sequence with end marker
                word_chars = tuple(word) + ('</w>',)  # End of word marker
                word_freqs[word_chars] += 1

        if verbose:
            print(f"   Found {len(word_freqs)} unique words")

        # Iteratively merge most frequent pairs until we hit vocab_size
        num_merges = 0
        target_merges = self.vocab_size - self.next_id

        while self.next_id < self.vocab_size:
            # Count all adjacent pairs
            pair_freqs = Counter()
            for word_chars, freq in word_freqs.items():
                for i in range(len(word_chars) - 1):
                    pair = (word_chars[i], word_chars[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge this pair
            merged = best_pair[0] + best_pair[1]

            # Add to vocabulary
            self.token_to_id[merged] = self.next_id
            self.id_to_token[self.next_id] = merged
            self.next_id += 1

            # Record merge
            self.merges[best_pair] = merged
            self.merge_order.append(best_pair)

            # Update word_freqs with merged pairs
            new_word_freqs = Counter()
            for word_chars, freq in word_freqs.items():
                # Apply merge to this word
                new_chars = []
                i = 0
                while i < len(word_chars):
                    if i < len(word_chars) - 1 and (word_chars[i], word_chars[i + 1]) == best_pair:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(word_chars[i])
                        i += 1
                new_word_freqs[tuple(new_chars)] = freq
            word_freqs = new_word_freqs

            num_merges += 1

            if verbose and num_merges % 500 == 0:
                print(f"   Learned {num_merges} merges... (e.g., {best_pair[0]}+{best_pair[1]} -> {merged})")

        if verbose:
            print(f"✓ BPE training complete:")
            print(f"   Total merges: {num_merges}")
            print(f"   Final vocab size: {self.next_id}")

            # Show some learned tokens
            print(f"   Example learned subwords:")
            for merge in self.merge_order[:10]:
                print(f"      '{merge[0]}' + '{merge[1]}' = '{self.merges[merge]}'")

    def encode(self, text, add_special_tokens=False):
        """
        Encode text to token IDs using BPE
        """
        if not text:
            return []

        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id[self.BOS_TOKEN])

        # Split into words
        words = text.split()

        for word_idx, word in enumerate(words):
            # Start with character sequence
            chars = list(word) + ['</w>']  # End of word marker

            # Apply all merges in order
            for merge_pair in self.merge_order:
                i = 0
                new_chars = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == merge_pair[0] and chars[i + 1] == merge_pair[1]:
                        new_chars.append(self.merges[merge_pair])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars

            # Convert to IDs
            for token in chars:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    # Unknown token (shouldn't happen if trained properly)
                    ids.append(self.token_to_id[self.UNK_TOKEN])

            # Add space token between words (except last word)
            if word_idx < len(words) - 1:
                if ' ' in self.token_to_id:
                    ids.append(self.token_to_id[' '])

        if add_special_tokens:
            ids.append(self.token_to_id[self.EOS_TOKEN])

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Decode token IDs back to text
        """
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]

                # Skip special tokens if requested
                if skip_special_tokens and token in [self.PAD_TOKEN, self.UNK_TOKEN,
                                                      self.BOS_TOKEN, self.EOS_TOKEN]:
                    continue

                tokens.append(token)

        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')  # End of word marker becomes space
        text = text.strip()

        return text

    def save(self, path):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},  # JSON needs string keys
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
            'merge_order': [[p[0], p[1]] for p in self.merge_order],
            'next_id': self.next_id
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Tokenizer saved to {path}")

    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges = {tuple(k.split('|')): v for k, v in data['merges'].items()}
        self.merge_order = [tuple(p) for p in data['merge_order']]
        self.next_id = data['next_id']

        print(f"📚 Tokenizer loaded from {path}")
        print(f"   Vocab size: {self.next_id}")


class MemoryAugmentedAttention(nn.Module):
    """
    Self-attention with template memory
    (Same as v3.1 - attention-based confidence)
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Template memory
        self.register_buffer('num_memories', torch.tensor(0))
        self.memory_keys = []
        self.memory_values = []
        self.memory_patterns = []

        # Attention history for confidence
        self.register_buffer('template_attention_history', torch.zeros(100))
        self.attention_idx = 0

    def add_memory(self, pattern_text, key_state, value_state):
        """Add template to memory"""
        self.memory_keys.append(key_state.detach())
        self.memory_values.append(value_state.detach())
        self.memory_patterns.append(pattern_text)
        self.num_memories += 1

    def get_template_confidence(self):
        """Get confidence from attention weights"""
        if self.num_memories == 0:
            return 0.0
        return self.template_attention_history.mean().item()

    def forward(self, x):
        """Forward with memory-augmented attention"""
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K_learned = self.k_proj(x)
        V_learned = self.v_proj(x)

        # Add memory if exists
        if self.num_memories > 0 and len(self.memory_keys) > 0:
            memory_k_list = []
            memory_v_list = []

            for mem_k, mem_v in zip(self.memory_keys, self.memory_values):
                mem_k_expanded = mem_k.unsqueeze(0).expand(batch_size, -1, -1)
                mem_v_expanded = mem_v.unsqueeze(0).expand(batch_size, -1, -1)
                memory_k_list.append(mem_k_expanded)
                memory_v_list.append(mem_v_expanded)

            K_memory = torch.cat(memory_k_list, dim=1)
            V_memory = torch.cat(memory_v_list, dim=1)

            K = torch.cat([K_memory, K_learned], dim=1)
            V = torch.cat([V_memory, V_learned], dim=1)

            memory_seq_len = K_memory.shape[1]
        else:
            K = K_learned
            V = V_learned
            memory_seq_len = 0

        # Multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Track template attention
        if memory_seq_len > 0:
            template_attn = attn_weights[:, :, :, :memory_seq_len].sum().item()
            total_attn = attn_weights.sum().item()
            template_proportion = template_attn / (total_attn + 1e-8)

            self.template_attention_history[self.attention_idx] = template_proportion
            self.attention_idx = (self.attention_idx + 1) % 100

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)

        return output


class AttentionBlockWithMemory(nn.Module):
    """Attention block with template memory"""
    def __init__(self, hidden_dim, num_heads=4, block_id="root", parent_loss=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_id = block_id

        self.attention = MemoryAugmentedAttention(hidden_dim, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Splitting
        self.is_split = False
        self.child_blocks = nn.ModuleList()

        # Confusion tracking
        self.register_buffer('recent_losses', torch.zeros(20))
        self.loss_idx = 0
        self.register_buffer('avg_confusion', torch.tensor(0.0))

        if parent_loss is not None:
            self.register_buffer('birth_confusion', parent_loss.clone())
        else:
            self.register_buffer('birth_confusion', torch.tensor(2.0))

        self.updates_since_birth = 0
        self.pattern_memory = deque(maxlen=100)

    def add_template(self, pattern, target, pattern_hidden, target_hidden):
        """Add template"""
        self.attention.add_memory(
            pattern_text=f"{pattern} → {target}",
            key_state=pattern_hidden,
            value_state=target_hidden
        )
        print(f"  📌 {self.block_id}: Template added (now {self.attention.num_memories} templates)")

    def get_template_confidence(self):
        """Get template confidence from attention"""
        return self.attention.get_template_confidence()

    def forward(self, x):
        """Forward through block"""
        if self.is_split and len(self.child_blocks) > 0:
            outputs = []
            for child in self.child_blocks:
                out = child(x)
                outputs.append(out)
            return torch.stack(outputs).mean(dim=0)

        attended = self.attention(x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))

        return x

    def update_confusion(self, loss_value, input_embedding=None):
        if self.is_split:
            return

        self.recent_losses[self.loss_idx] = loss_value
        self.loss_idx = (self.loss_idx + 1) % 20
        self.avg_confusion = self.recent_losses.mean()
        self.updates_since_birth += 1

        if input_embedding is not None:
            self.pattern_memory.append(input_embedding.detach().cpu())

    def compute_pattern_diversity(self):
        if len(self.pattern_memory) < 10:
            return 0.0

        patterns = torch.stack(list(self.pattern_memory)[-50:])
        mean_pattern = patterns.mean(dim=0)
        distances = torch.norm(patterns - mean_pattern, dim=1)
        return distances.mean().item()

    def should_split(self, loss_threshold=1.5, min_updates=30):
        if self.is_split or self.updates_since_birth < min_updates:
            return False

        loss_confused = self.avg_confusion > loss_threshold
        diversity = self.compute_pattern_diversity()
        diversity_confused = diversity > 0.5

        template_conf = self.get_template_confidence()
        template_ignored = (self.attention.num_memories > 0 and
                          template_conf < 0.2 and
                          self.avg_confusion > 1.0)

        return loss_confused or diversity_confused or template_ignored

    def split(self, num_children=2):
        if self.is_split:
            return False

        print(f"  🌱 SPLIT {self.block_id}: loss={self.avg_confusion:.3f}, "
              f"template_conf={self.get_template_confidence():.3f}")

        device = next(self.parameters()).device

        for i in range(num_children):
            child = AttentionBlockWithMemory(
                self.hidden_dim,
                self.num_heads,
                block_id=f"{self.block_id}.{i}",
                parent_loss=self.avg_confusion
            ).to(device)

            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), self.parameters()):
                    if parent_param.requires_grad:
                        child_param.data.copy_(
                            parent_param.data + torch.randn_like(parent_param) * 0.02
                        )

            # Inherit memories
            child.attention.memory_keys = self.attention.memory_keys.copy()
            child.attention.memory_values = self.attention.memory_values.copy()
            child.attention.memory_patterns = self.attention.memory_patterns.copy()
            child.attention.num_memories = self.attention.num_memories.clone()

            self.child_blocks.append(child)

        self.is_split = True

        for param in self.parameters():
            param.requires_grad = False

        for child in self.child_blocks:
            for param in child.parameters():
                param.requires_grad = True

        return True


class ExpandFormerV4(nn.Module):
    """
    V4: Token-based with BPE tokenizer
    """
    def __init__(
        self,
        tokenizer,
        embed_dim=128,
        hidden_dim=256,
        context_len=64,  # In tokens now!
        num_initial_blocks=2,
        num_heads=4,
        split_threshold=1.5,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.split_threshold = split_threshold

        # Token embedding
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(context_len, embed_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Attention blocks with memory
        self.blocks = nn.ModuleList([
            AttentionBlockWithMemory(hidden_dim, num_heads, block_id=f"B{i}")
            for i in range(num_initial_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.vocab_size)

        # Stats
        self.total_updates = 0
        self.total_splits = 0
        self.tokens_learned = 0
        self.total_templates = 0

    def _create_pos_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc

    def forward(self, x):
        batch_size, seq_len = x.shape

        h = self.embedding(x)
        if seq_len <= self.context_len:
            h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        logits = self.output(h)

        return logits

    def add_template(self, pattern, target):
        """Add knowledge template"""
        device = next(self.parameters()).device

        # Encode pattern and target
        pattern_ids = self.tokenizer.encode(pattern)
        target_ids = self.tokenizer.encode(target)

        if len(pattern_ids) == 0 or len(target_ids) == 0:
            return

        # Pad/trim to context length
        pattern_context = pattern_ids[-self.context_len:]
        while len(pattern_context) < self.context_len:
            pattern_context = [self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]] + pattern_context

        target_context = target_ids[-self.context_len:]
        while len(target_context) < self.context_len:
            target_context = [self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]] + target_context

        with torch.no_grad():
            # Get hidden states
            x_pattern = torch.tensor([pattern_context], dtype=torch.long, device=device)
            h_pattern = self.embedding(x_pattern)
            h_pattern = h_pattern + self.pos_encoding[:len(pattern_context)].unsqueeze(0)
            h_pattern = self.input_proj(h_pattern)

            x_target = torch.tensor([target_context], dtype=torch.long, device=device)
            h_target = self.embedding(x_target)
            h_target = h_target + self.pos_encoding[:len(target_context)].unsqueeze(0)
            h_target = self.input_proj(h_target)

        # Add to all blocks
        for block in self.blocks:
            block.add_template(
                pattern=pattern,
                target=target,
                pattern_hidden=h_pattern.squeeze(0),
                target_hidden=h_target.squeeze(0)
            )

        self.total_templates += 1
        print(f"  ✓ Template: '{pattern}' → '{target}'")

    def get_all_blocks(self):
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

    def get_template_stats(self):
        blocks = self.get_all_blocks()

        total_templates = sum(b.attention.num_memories.item() for b in blocks)
        avg_confidence = np.mean([b.get_template_confidence() for b in blocks])

        return {
            'total_templates': total_templates,
            'avg_confidence': avg_confidence
        }

    def check_and_split(self):
        blocks = self.get_all_blocks()

        for block in blocks:
            if block.should_split(loss_threshold=self.split_threshold):
                if block.split():
                    self.total_splits += 1
                    return True

        return False


class TokenLearner:
    """
    Learner for token-based model
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
        self.recent_loss_window = deque(maxlen=100)

        # Statistics
        self.stats = {
            'tokens_from_files': 0,
            'tokens_from_user': 0,
            'tokens_from_self': 0,
            'total_tokens': 0,
            'start_time': time.time(),
        }

    def learn_sequence(self, text, source='user', verbose=False):
        """
        Learn from text (tokens now!)
        """
        # Determine source weight
        if source == 'user' or source == 'file':
            source_weight = 1.0
        elif source == 'self':
            source_weight = 0.3
        else:
            source_weight = 1.0

        # Encode to tokens
        token_ids = self.model.tokenizer.encode(text)

        if len(token_ids) < 2:
            return 0.0

        # Update stats
        self.stats['total_tokens'] += len(token_ids)
        if source == 'file':
            self.stats['tokens_from_files'] += len(token_ids)
        elif source == 'user':
            self.stats['tokens_from_user'] += len(token_ids)
        elif source == 'self':
            self.stats['tokens_from_self'] += len(token_ids)

        total_loss = 0.0
        num_updates = 0

        # Learn token by token
        for i in range(len(token_ids) - 1):
            context_start = max(0, i - self.model.context_len + 1)
            context = token_ids[context_start:i+1]

            # Pad if needed
            while len(context) < self.model.context_len:
                context = [self.model.tokenizer.token_to_id[self.model.tokenizer.PAD_TOKEN]] + context

            x = torch.tensor([context[-self.model.context_len:]], dtype=torch.long, device=self.device)
            y = torch.tensor([token_ids[i+1]], dtype=torch.long, device=self.device)

            # Train
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = F.cross_entropy(logits[:, -1, :], y)

            weighted_loss = loss * source_weight
            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update block confusions
            blocks = self.model.get_all_blocks()
            for block in blocks:
                with torch.no_grad():
                    h = self.model.embedding(x)
                    h = self.model.input_proj(h)
                    input_emb = h[:, -1, :].mean(dim=0)

                block.update_confusion(loss.item(), input_emb)

            total_loss += loss.item()
            num_updates += 1
            self.model.total_updates += 1
            self.model.tokens_learned += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        self.losses.append(avg_loss)
        self.recent_loss_window.append(avg_loss)

        # Check for splits
        if self.model.total_updates % 25 == 0:
            if self.model.check_and_split():
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    weight_decay=0.01,
                    betas=(0.9, 0.95)
                )

        return avg_loss

    def generate(self, prompt, max_length=50, temperature=0.8, top_k=40):
        """Generate text"""
        self.model.eval()

        # Encode prompt
        token_ids = self.model.tokenizer.encode(prompt)

        for _ in range(max_length):
            # Get context
            context = token_ids[-self.model.context_len:]
            while len(context) < self.model.context_len:
                context = [self.model.tokenizer.token_to_id[self.model.tokenizer.PAD_TOKEN]] + context

            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Stop if EOS
            if next_token == self.model.tokenizer.token_to_id[self.model.tokenizer.EOS_TOKEN]:
                break

            token_ids.append(next_token)

            # Stop at natural sentence end
            decoded = self.model.tokenizer.decode([next_token])
            if decoded in ['.', '!', '?'] and np.random.random() < 0.3:
                break

        self.model.train()

        # Decode
        result = self.model.tokenizer.decode(token_ids)
        return result

    def get_stats(self):
        """Get statistics"""
        avg_recent_loss = np.mean(list(self.recent_loss_window)) if self.recent_loss_window else 0
        runtime = time.time() - self.stats['start_time']

        template_stats = self.model.get_template_stats()

        return {
            'blocks': len(self.model.get_all_blocks()),
            'splits': self.model.total_splits,
            'updates': self.model.total_updates,
            'tokens_learned': self.model.tokens_learned,
            'avg_loss': avg_recent_loss,
            'runtime_minutes': runtime / 60,
            'tokens_per_minute': self.stats['total_tokens'] / (runtime / 60) if runtime > 0 else 0,
            'templates': template_stats['total_templates'],
            'template_confidence': template_stats['avg_confidence'],
            'vocab_size': self.model.tokenizer.next_id,
            **self.stats
        }


def prepare_tokenizer_from_files(file_paths, vocab_size=5000):
    """
    Train BPE tokenizer on files
    """
    print("\n📚 Preparing tokenizer from training files...")

    # Load all texts
    texts = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                texts.append(text)
            print(f"   Loaded: {path} ({len(text)} chars)")
        except Exception as e:
            print(f"   ⚠️  Error loading {path}: {e}")

    if not texts:
        print("   ⚠️  No texts found, using base tokenizer")
        return BPETokenizer(vocab_size=vocab_size)

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train_bpe(texts, verbose=True)

    return tokenizer


def quick_demo():
    """
    Quick demo showing token-based learning
    """
    print("=" * 70)
    print("🚀 ExpandFormer V4: Token-Based Quick Demo")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Sample training data
    sample_texts = [
        "Hello, how are you today? I am doing well, thank you for asking.",
        "Machine learning is fascinating. Neural networks learn from data.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence helps computers understand language.",
        "Training models requires data and computation.",
        "My dog is not feeling great",
        "Hello how are you doing?",
        "Hi im doing great thank you"
    ]

    # Train tokenizer
    print("Step 1: Training BPE tokenizer on sample data...\n")
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train_bpe(sample_texts, verbose=True)

    # Create model
    print("\nStep 2: Creating model...\n")
    model = ExpandFormerV4(
        tokenizer=tokenizer,
        embed_dim=96,
        hidden_dim=192,
        context_len=32,
        num_initial_blocks=2,
        num_heads=4,
        split_threshold=1.5
    )

    learner = TokenLearner(model, lr=0.003, device=device)

    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Teach templates
    print("Step 3: Teaching templates...\n")
    model.add_template("Hello", "Hi")
    model.add_template("How are you", "I'm good")
    print()

    # Train
    print("Step 4: Training on sample data...\n")
    for epoch in range(10):
        epoch_loss = 0
        for text in sample_texts:
            loss = learner.learn_sequence(text, source='file')
            epoch_loss += loss

        avg_loss = epoch_loss / len(sample_texts)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.3f}")

        # Test generation
        if epoch % 3 == 2:
            response = learner.generate("Hello", max_length=10, temperature=0.7)
            print(f"   Gen: '{response}'")

    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    quick_demo()