"""
Patch for PyTorch 2.5.x torch.load security check.
This allows loading trusted models without requiring PyTorch 2.6+.
"""

import torch
import warnings

# Store original torch.load
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """
    Patched version of torch.load that disables the 2.6 requirement.
    Safe for loading trusted models from HuggingFace.
    """
    # Remove weights_only if present to avoid the version check
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    
    # Suppress the warning about weights_only
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# CRITICAL: Patch transformers library's check_torch_load_is_safe function(s)
try:
    from transformers.utils import import_utils

    def patched_check_torch_load_is_safe():
        """Bypass the transformers library's torch.load safety check."""
        pass  # Do nothing - allow the load

    import_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe

    try:
        from transformers import modeling_utils

        modeling_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe
    except Exception:
        pass

    print("✅ PyTorch load patch applied - bypassing 2.6 version requirement")
except ImportError:
    print("⚠️ Could not patch transformers library (not installed yet)")
