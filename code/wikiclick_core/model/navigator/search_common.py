"""Budget accounting shared by zero-shot graph searches. No oracle labels."""
import time
import numpy as np

BUDGETS=(25,50,100,200)


class Trace:
    def __init__(self,budgets):
        self.budgets=tuple(sorted(set(budgets)))
        if not self.budgets or self.budgets[0]<1:raise ValueError('Budgets must be positive')
        self.started=time.perf_counter();self.expanded=0;self.unique=set();self.scored=0;self.peak=1;self.records={}

    def record(self,path,status):
        return {'path':list(path) if path is not None else None,'route_clicks':len(path)-1 if path is not None else None,
            'expanded':self.expanded,'unique_expanded':len(self.unique),'candidates_scored':self.scored,
            'seconds':time.perf_counter()-self.started,'peak_frontier_size':self.peak,'status':status}

    def checkpoint(self):
        if self.expanded in self.budgets:self.records[str(self.expanded)]=self.record(None,'expansion_budget')

    def finish(self,path,status):
        final=self.record(path,status)
        for budget in self.budgets:
            if str(budget) not in self.records:self.records[str(budget)]=final.copy()
        return self.records


def scores(policy,current,candidates,target,trace):
    values=policy.score(current,candidates,target).detach().cpu().numpy()
    if values.shape!=(len(candidates),) or not np.isfinite(values).all():raise ValueError('Invalid policy scores')
    trace.scored+=len(candidates)
    return values
