"""Z2: depth-synchronous beam search ranked by endpoint cosine, not probabilities."""
from .search_common import Trace,scores,BUDGETS


def navigate(graph,source,target,policy,width=8,budgets=BUDGETS,max_clicks=20):
    if width<1:raise ValueError('Beam width must be positive')
    trace=Trace(budgets);beam=[(source,)]
    if source==target:return trace.finish([source],'found')
    for depth in range(max_clicks):
        generated={}
        for offset,path in enumerate(beam):
            if trace.expanded>=max(trace.budgets):return trace.finish(None,'expansion_budget')
            current=path[-1];trace.expanded+=1;trace.unique.add(current)
            candidates=sorted(n for n in graph.successors(current) if n not in path)
            if target in candidates:return trace.finish(path+(target,),'found')
            values=scores(policy,current,candidates,target,trace) if candidates else []
            if depth+1<max_clicks:
                for n,value in zip(candidates,values):
                    next_path=path+(n,)
                    # At each depth keep one deterministic prefix per endpoint.
                    if n not in generated or next_path<generated[n][1]:generated[n]=(float(value),next_path)
            trace.peak=max(trace.peak,len(beam)-offset-1+len(generated));trace.checkpoint()
        if not generated:return trace.finish(None,'click_limit' if depth+1>=max_clicks else 'frontier_exhausted')
        beam=[path for _,path in sorted(generated.values(),key=lambda pair:(-pair[0],pair[1]))[:width]]
    return trace.finish(None,'click_limit')
