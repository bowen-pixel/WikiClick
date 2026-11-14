"""Z0: cycle-free local cosine navigation, with explicit expansion/click limits."""
from .search_common import Trace,scores,BUDGETS


def navigate(graph,source,target,policy,budgets=BUDGETS,max_clicks=20):
    trace=Trace(budgets);path=[source];seen={source}
    if source==target:return trace.finish(path,'found')
    while len(path)-1<max_clicks and trace.expanded<max(trace.budgets):
        current=path[-1];trace.expanded+=1;trace.unique.add(current)
        candidates=sorted(n for n in graph.successors(current) if n not in seen)
        if target in candidates:return trace.finish(path+[target],'found')
        if not candidates:return trace.finish(None,'dead_end')
        values=scores(policy,current,candidates,target,trace)
        chosen=min(zip(candidates,values),key=lambda pair:(-pair[1],pair[0]))[0]
        path.append(chosen);seen.add(chosen);trace.checkpoint()
    return trace.finish(None,'click_limit' if len(path)-1>=max_clicks else 'expansion_budget')
