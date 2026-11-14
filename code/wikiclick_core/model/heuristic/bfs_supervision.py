"""Reuse NetworkX reverse BFS; labels never form a model input."""
import networkx as nx
import numpy as np


def distances(graph,page_ids,goals):
    values=np.full((len(goals),len(page_ids)),-1,dtype=np.int16)
    for i,goal in enumerate(goals):
        d=nx.single_source_shortest_path_length(graph.reverse(copy=False),goal)
        if max(d.values(),default=0)>=32767:raise ValueError('Distance storage overflow')
        values[i]=[d.get(n,-1) for n in page_ids]
    return values


def optimal_mask(current,candidates,goal_index,oracle):
    d=int(oracle[goal_index,current])
    if d<1:raise ValueError('Training current must be reachable and nonterminal')
    return oracle[goal_index,candidates]==d-1


def progress_targets(candidate_distances):
    d=np.asarray(candidate_distances)
    return np.where(d>=0,1/(1+np.maximum(d,0)),0).astype(np.float32)
