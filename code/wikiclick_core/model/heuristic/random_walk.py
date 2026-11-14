"""Offline forward-walk examples; live inference integration is future work."""
import random
import time
import numpy as np
import networkx as nx
LENGTHS=[3,5,10,20]

def random_forward(graph,rng,length):
    path=[rng.choice(sorted(graph))]
    for _ in range(length):
        neighbors=sorted(graph.successors(path[-1]))
        if not neighbors:break
        path.append(rng.choice(neighbors))
    return path


def terminal_prefix(path):
    """Episode stops on first encounter with its eventual endpoint goal."""
    return path[:path.index(path[-1])+1]


def examples(graph,path,walk_id,kind):
    goal=path[-1];rows=[]
    for current,action in zip(path,path[1:]):
        candidates=sorted(graph.successors(current))
        if action not in candidates or current==goal:raise ValueError('Invalid terminalized demonstration')
        rows.append({'current':current,'goal':goal,'candidates':candidates,'action':action,'walk_id':walk_id,'kind':kind})
    return rows


def generate(graph,training_goals,action_budget=4096,seed=963):
    if action_budget<2 or action_budget%2:raise ValueError('Use a positive even action budget')
    rng=random.Random(seed);allowed=set(training_goals);walks=[];forward=[];optimal=[]
    rejected=0;empty=0;attempts=0;walk_seconds=0.;forward_seconds=0.;shortest_seconds=0.;started=time.perf_counter()
    # One shared endpoint-pair pool, large enough to sample the same number of
    # actions from both supervision methods without silently duplicating rows.
    while min(len(forward),len(optimal))<2*action_budget:
        attempts+=1
        if attempts>200000:raise ValueError('Insufficient accepted trajectories')
        requested=LENGTHS[(attempts-1)%len(LENGTHS)]
        before=time.perf_counter();path=random_forward(graph,rng,requested);walk_seconds+=time.perf_counter()-before
        if path[-1] not in allowed:rejected+=1;continue
        prefix=terminal_prefix(path)
        if len(prefix)<2:empty+=1;continue
        walk_id=len(walks)
        before=time.perf_counter();f=examples(graph,prefix,walk_id,'random_forward');forward_seconds+=time.perf_counter()-before
        before=time.perf_counter();shortest=nx.shortest_path(graph,prefix[0],prefix[-1]);o=examples(graph,shortest,walk_id,'shortest_path');shortest_seconds+=time.perf_counter()-before
        walks.append({'walk_id':walk_id,'requested_steps':requested,'actual_steps':len(path)-1,'path':path,
            'terminalized_path':prefix,'first_goal_step':len(prefix)-1,'repeated_vertex_visits':len(path)-len(set(path)),
            'early_goal_encounters':sum(n==path[-1] for n in path[:-1]),'shortest_path':shortest})
        forward.extend(f);optimal.extend(o)
    indices=rng.sample(range(len(forward)),action_budget);b1=[forward[i] for i in indices]
    by_goal={}
    for row in optimal:by_goal.setdefault(row['goal'],[]).append(row)
    required={}
    for row in b1:required[row['goal']]=required.get(row['goal'],0)+1
    # Matching goal histograms can exceed a shortest-path pool's local count;
    # sampling with replacement is explicit and its unique-action count is logged.
    selected={g:rng.choices(by_goal[g],k=count) for g,count in required.items()}
    b2=[selected[r['goal']].pop() for r in b1]
    mask=set(rng.sample(range(action_budget),action_budget//2))
    b3=[b1[i] if i in mask else b2[i] for i in range(action_budget)]
    arms={'B1':b1,'B2':b2,'B3':b3}
    for rows in arms.values():
        assert len(rows)==action_budget and all(r['goal'] in allowed for r in rows)
        assert {g:sum(r['goal']==g for r in rows) for g in required}==required
    stats={'attempted_walks':attempts,'accepted_walks':len(walks),'reserved_goal_rejections':rejected,'zero_action_terminal_prefix_rejections':empty,
        'lengths':{str(n):{'walks':sum(w['requested_steps']==n for w in walks),
            'mean_actual_steps':float(np.mean([w['actual_steps'] for w in walks if w['requested_steps']==n])) if any(w['requested_steps']==n for w in walks) else None,
            'mean_retained_actions':float(np.mean([len(w['terminalized_path'])-1 for w in walks if w['requested_steps']==n])) if any(w['requested_steps']==n for w in walks) else None} for n in LENGTHS},
        'walks_with_repeated_vertices':sum(w['repeated_vertex_visits']>0 for w in walks),
        'walks_with_early_goal_encounters':sum(w['early_goal_encounters']>0 for w in walks),
        'pool_forward_actions':len(forward),'pool_shortest_actions':len(optimal),'goal_action_histogram':required,
        'training_action_budget':action_budget,'training_goals_observed':len(required),'requested_training_goals':len(allowed),
        'unique_state_goal_action_triples':{a:len({(r['current'],r['goal'],r['action']) for r in rows}) for a,rows in arms.items()},
        'mixed_forward_actions':sum(r['kind']=='random_forward' for r in b3),
        'timing':{'shared_random_walk_generation_seconds':walk_seconds,'forward_action_conversion_seconds':forward_seconds,
            'paired_shortest_path_and_conversion_seconds':shortest_seconds,'total_generation_seconds':time.perf_counter()-started},
        'sampling_note':'B1 samples forward action records without replacement; B2 samples paired-pool shortest actions with replacement to match B1 goal counts exactly; B3 substitutes exactly half the aligned B1/B2 records, preserving the same goal histogram',
        'terminal_note':'Walks allow repeats. Endpoint chooses goal; supervision stops at its first occurrence, and zero-action cycles are discarded. This is an explicit episodic modification to raw fixed-length forward-walk cloning.'}
    return arms,walks,stats
