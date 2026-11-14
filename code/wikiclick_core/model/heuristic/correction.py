"""Offline training-mistake resampling; live navigation never updates weights."""
import torch


def mistake_weights(count, mistakes, multiplier=4.):
    if multiplier <= 0:
        raise ValueError("Positive resampling weight required")
    weights = torch.ones(count)
    weights[mistakes] = multiplier
    return weights
