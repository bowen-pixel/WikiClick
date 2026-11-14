"""Deterministic signed SHA256 title features; hashing is not semantic learning."""
import hashlib
import re
import numpy as np
import torch

def title_features(title, dimensions=128):
    """Original, deterministic signed SHA256 word/character hashing."""
    if dimensions < 1:
        raise ValueError("Feature dimension must be positive")
    text = " " + " ".join(title.lower().split()) + " "
    tokens = re.findall(r"\w+", text)
    tokens += [text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1))]
    result = torch.zeros(dimensions)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "little") % dimensions] += 1 if digest[4] % 2 else -1
    return result / result.norm().clamp_min(1)

def fast_title_features(title, dimensions=128):
    """Equivalent fixed hashing; count in NumPy rather than many tensor updates."""
    text = " "+" ".join(title.lower().split())+" "
    tokens = re.findall(r"\w+", text)
    tokens += [text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1))]
    indices, signs = [], []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        indices.append(int.from_bytes(digest[:4], "little") % dimensions)
        signs.append(1 if digest[4] % 2 else -1)
    counts = np.bincount(indices, weights=signs, minlength=dimensions).astype(np.float32)
    result = torch.from_numpy(counts)
    return (result/result.norm().clamp_min(1)).numpy()
