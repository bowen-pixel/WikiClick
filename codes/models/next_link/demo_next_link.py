"""Explain one real next-link training step using the saved 1000-article baseline.

Runs offline. Changes weights only in memory; never writes a checkpoint or affects
the running million-node workflow. This is an illustration, not a new benchmark.
"""
import copy
import json
from pathlib import Path

import torch

from codes.models.raw.ml_model import pair_features
from codes.models.next_link.next_link_experiment import Ranker, ranking_loss


def main():
    torch.set_num_threads(2)
    folder = Path("models/next_link/ranking_1000")
    checkpoint = torch.load(folder/"seed_207/raw.pt",map_location="cpu",weights_only=True)
    features = torch.load(folder/"features.pt",map_location="cpu",weights_only=True)
    graph = json.loads(Path("data/graphs/ranking_1000/graph.json").read_text())
    if checkpoint["graph_id"] != features["graph_id"] or checkpoint["graph_id"] != graph["graph_id"]:
        raise ValueError("Graph, features, and checkpoint differ")
    titles = {n["pageid"]:n["title"] for n in graph["nodes"]}
    names = [titles[pageid] for pageid in features["page_ids"]]
    rows = json.loads(Path("data/graphs/ranking_1000/train_decisions.json").read_text())
    model = Ranker(checkpoint["state_dict"])
    title = features["title_vectors"]
    semantic = features["semantic_vectors"]  # Unused by this title-only demonstration.

    # Choose a TRAINING mistake for illustration, never a test query.
    row = rows[0]
    with torch.inference_mode():
        for candidate_row in rows[:200]:
            c = torch.tensor(candidate_row["candidates"])
            d = torch.full_like(c,candidate_row["goal"])
            scores = model(c,d,title,semantic)
            if not candidate_row["optimal"][int(scores.argmax())]:
                row = candidate_row
                break
    candidates = torch.tensor(row["candidates"])
    destinations = torch.full_like(candidates,row["goal"])
    optimal = torch.tensor(row["optimal"],dtype=torch.bool)
    print("Current article:",names[row["current"]])
    print("Destination:",names[row["goal"]])
    print("Outgoing candidates:",len(candidates),"| Optimal choices:",int(optimal.sum()))
    print("Pair-feature shape:",tuple(pair_features(title[candidates],title[destinations]).shape))
    print("Optimal next links:",", ".join(names[int(b)] for b in candidates[optimal]))

    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(),lr=.0003)
    model.train()
    logits = model(candidates,destinations,title,semantic)
    loss = ranking_loss(logits,row["optimal"])
    before_loss = loss.item()
    before_choice = int(logits.argmax())
    before_mass = logits.softmax(0)[optimal].sum().item()

    optimizer.zero_grad()
    loss.backward()
    gradient_norm = sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None)**.5
    optimizer.step()

    with torch.inference_mode():
        after_logits = model(candidates,destinations,title,semantic)
        after_loss = ranking_loss(after_logits,row["optimal"]).item()
        after_choice = int(after_logits.argmax())
        after_mass = after_logits.softmax(0)[optimal].sum().item()
    change = sum((model.state_dict()[k]-v).square().sum().item() for k,v in initial.items())**.5
    print(f"Ranking loss: {before_loss:.6f} -> {after_loss:.6f}")
    print(f"Probability on optimal choices: {before_mass:.4f} -> {after_mass:.4f}")
    print("Top choice before:",names[int(candidates[before_choice])],"| optimal:",bool(optimal[before_choice]))
    print("Top choice after: ",names[int(candidates[after_choice])],"| optimal:",bool(optimal[after_choice]))
    print(f"Gradient L2: {gradient_norm:.6f} | Parameter change L2: {change:.6f}")
    print("Only one in-memory training step. No checkpoint changed; not an improvement benchmark.")


if __name__ == "__main__":
    main()
