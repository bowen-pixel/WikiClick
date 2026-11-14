"""Z1 semantic best-first and Z3 FIFO BFS; returned routes are offline plans."""
from collections import deque
import heapq
import itertools
from .search_common import Trace,scores,BUDGETS


def navigate(graph,source,target,policy=None,budgets=BUDGETS,max_clicks=20):
    trace=Trace(budgets);counter=itertools.count()
    queue=deque([(source,)]) if policy is None else [(0.,next(counter),(source,))]
    best_depth={source:0};expanded_depth={};depth_limited=False
    if source==target:return trace.finish([source],'found')
    while queue:
        if trace.expanded>=max(trace.budgets):return trace.finish(None,'expansion_budget')
        path=queue.popleft() if policy is None else heapq.heappop(queue)[2]
        current=path[-1];depth=len(path)-1
        if depth>best_depth[current] or expanded_depth.get(current,float('inf'))<=depth:continue
        if depth>=max_clicks:depth_limited=True;continue
        expanded_depth[current]=depth;trace.expanded+=1;trace.unique.add(current)
        candidates=sorted(n for n in graph.successors(current) if n not in path)
        if target in candidates:return trace.finish(path+(target,),'found')
        values=scores(policy,current,candidates,target,trace) if policy is not None and candidates else [0.]*len(candidates)
        for n,value in zip(candidates,values):
            next_depth=depth+1
            if next_depth>=max_clicks:depth_limited=True;continue
            if next_depth>=best_depth.get(n,float('inf')):continue
            best_depth[n]=next_depth;next_path=path+(n,)
            if policy is None:queue.append(next_path)
            else:heapq.heappush(queue,(-float(value),next(counter),next_path))
        trace.peak=max(trace.peak,len(queue));trace.checkpoint()
    return trace.finish(None,'click_limit' if depth_limited else 'frontier_exhausted')
