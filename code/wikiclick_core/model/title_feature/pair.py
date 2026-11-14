"""Candidate/goal interaction features shared by both representations."""
import torch

def pair_features(candidate, destination):
    """Same four feature blocks in BOTH methods; no distance/ID inputs."""
    return torch.cat((candidate, destination, (candidate-destination).abs(), candidate*destination), dim=-1)
