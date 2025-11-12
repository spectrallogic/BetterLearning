"""
ExpandFormer Standard Interface
================================

SUPER SIMPLE RULES:
1. Export a class called "ExpandFormer"
2. It has __init__(vocab_size, context_len, **kwargs)
3. It has forward(x) that returns logits
4. Everything else is OPTIONAL

That's it. No inheritance. No abstract methods. Total freedom.

EXAMPLE:

# expandformer_v99.py
class MyCrazyArchitecture(nn.Module):
    def __init__(self, vocab_size, context_len, **kwargs):
        super().__init__()
        # Do whatever you want

    def forward(self, x):
        # Do whatever you want
        return logits

# Export it
ExpandFormer = MyCrazyArchitecture


OPTIONAL METHODS (for growth tracking):
- update_tracking(loss_value, token_id)  # Track difficulties
- check_and_grow()                        # Trigger growth
- get_model_info()                        # Return metadata dict

If your model has these, they'll be called. If not, no problem.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


# ============================================================================
# OPTIONAL HELPER FUNCTIONS
# ============================================================================

def has_method(obj, method_name: str) -> bool:
    """Check if object has a callable method"""
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))


def safe_call(obj, method_name: str, *args, **kwargs):
    """Call method if it exists, otherwise do nothing"""
    if has_method(obj, method_name):
        return getattr(obj, method_name)(*args, **kwargs)
    return None


def get_model_capabilities(model) -> Dict[str, bool]:
    """
    Check what optional features a model supports
    """
    return {
        'has_update_tracking': has_method(model, 'update_tracking'),
        'has_growth': has_method(model, 'check_and_grow'),
        'has_info': has_method(model, 'get_model_info'),
    }


# ============================================================================
# OPTIONAL WRAPPER (Makes ANY model compatible)
# ============================================================================

class ExpandFormerWrapper(nn.Module):
    """
    Wrap ANY PyTorch model to make it compatible with benchmark

    Usage:
        my_model = SomeWeirdArchitecture()
        wrapped = ExpandFormerWrapper(my_model)
        # Now wrapped has standard interface
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        # Extract if available
        self.vocab_size = getattr(model, 'vocab_size', None)
        self.context_len = getattr(model, 'context_len', None)

    def forward(self, x):
        return self.model(x)

    def update_tracking(self, loss_value: float, token_id: int):
        safe_call(self.model, 'update_tracking', loss_value, token_id)

    def check_and_grow(self) -> bool:
        result = safe_call(self.model, 'check_and_grow')
        return result if result is not None else False

    def get_model_info(self) -> Dict[str, Any]:
        if has_method(self.model, 'get_model_info'):
            return self.model.get_model_info()

        # Default info
        return {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


# ============================================================================
# VALIDATION (OPTIONAL - just checks conventions)
# ============================================================================

def validate_interface(model_class) -> tuple[bool, str]:
    """
    Check if class follows conventions (doesn't enforce them)
    Returns: (is_valid, message)
    """
    import inspect

    # Check if it's a class
    if not inspect.isclass(model_class):
        return False, "Not a class"

    # Check __init__ signature
    try:
        sig = inspect.signature(model_class.__init__)
        params = list(sig.parameters.keys())

        if 'vocab_size' not in params:
            return False, "Missing vocab_size parameter in __init__"

        if 'context_len' not in params:
            return False, "Missing context_len parameter in __init__"

    except Exception as e:
        return False, f"Could not inspect __init__: {e}"

    # Check forward method exists
    if not hasattr(model_class, 'forward'):
        return False, "Missing forward() method"

    return True, "Follows interface conventions"


# ============================================================================
# EXAMPLE/TEMPLATE
# ============================================================================

class MinimalExpandFormer(nn.Module):
    """
    Template showing MINIMUM requirements
    Copy this to start a new version!
    """

    def __init__(self, vocab_size, context_len, **kwargs):
        """
        REQUIRED SIGNATURE
        Accept vocab_size, context_len, and any other params via **kwargs
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len

        # Your architecture here
        self.embedding = nn.Embedding(vocab_size, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, batch_first=True),
            num_layers=2
        )
        self.output = nn.Linear(128, vocab_size)

    def forward(self, x):
        """
        REQUIRED METHOD
        Input: x [batch, seq_len] - token IDs
        Output: logits [batch, vocab_size] OR [batch, seq_len, vocab_size]
        """
        h = self.embedding(x)
        h = self.transformer(h)
        logits = self.output(h[:, -1, :])  # Last position only
        return logits

    # ========================================================================
    # OPTIONAL METHODS (only if your model needs them)
    # ========================================================================

    def update_tracking(self, loss_value: float, token_id: int):
        """OPTIONAL: Track for growth decisions"""
        pass

    def check_and_grow(self) -> bool:
        """OPTIONAL: Grow model if needed"""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """OPTIONAL: Return metadata"""
        return {
            'total_params': sum(p.numel() for p in self.parameters()),
        }


# ============================================================================
# EXPORT FOR USAGE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)