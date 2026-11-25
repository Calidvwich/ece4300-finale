"""Train a learned edit prior and evaluate novelty/recovery on held-out hard tasks."""
import json, hashlib, random, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CAUSAL=HERE.parents[1]
sys.path.insert(0,str(CAUSAL))
from evaluate_cached_ranker import features,sigmoid

DATA=CAUSAL/'benchmark_obs_count'/'n5_obs1to5_80each.json'
SAMPLES=CAUSAL/'benchmark_obs_count'/'saturation'
OUT=HERE/'graph_edit_report.json'
NODES=['A','B','C','D','E']
PAIRS=[(a,b) for a in NODES for b in NODES if a!=b]
RNG=random.Random(83102)

def h(g): return hashlib.sha256(json.dumps(sorted(map(tuple,g['edges'])),separators=(',',':')).encode()).hexdigest()

def dag(edges):
    adj={n:[] for n in NODES}
    for a,b in edges: adj[a].append(b)
    state={n:0 for n in NODES}
    def visit(n):
        if state[n]==1:return False
        if state[n]==2:return True
        state[n]=1
        if not all(visit(v) for v in adj[n]):return False
        state[n]=2;return True
    return all(visit(n) for n in NODES)

def variants(g, limit=8):
    edges=set(map(tuple,g['edges'])); out=[]
    choices=PAIRS[:]; RNG.shuffle(choices)
    for a,b in choices:
        e=set(edges)
        if (a,b) in e:e.remove((a,b))
        else:e.add((a,b))
        if dag(e): out.append({'nodes':NODES,'edges':sorted(e)})
        if len(out)>=limit:break
    return out

def validate(g,obs):
    if not dag(set(map(tuple,g['edges']))): return False
    # Reachability under intervention must match each observed descendant set.
    adj={n:[] for n in NODES}
    for a,b in g['edges']:adj[a].append(b)
    for o in obs:
        reached=set(); stack=list(adj[o['perturbed_node']])
        while stack:
            n=stack.pop()
            if n in reached:continue
            reached.add(n);stack.extend(adj[n])
        if any(int(n in reached)!=int(o['effects'].get(n,0)) for n in NODES):return False
    return True

def search(seed,obs,scorer,seed_value):
    RNG.seed(seed_value)
    found={h(seed)} if seed else set();frontier=[seed] if seed else [];queued={h(seed)} if seed else set();edits=0
    while frontier and edits<99:
        cand=[]
        for parent in frontier:
            for v in variants(parent,12):
                vh=h(v)
                if vh not in queued:cand.append((scorer(v,obs),v,vh));queued.add(vh)
        if not cand:break
        cand.sort(key=lambda x:(-x[0],x[2]))
        newfront=[]
        for _,g,gh in cand:
            if edits>=99:break
            edits+=1
            if validate(g,obs) and gh not in found:
                found.add(gh);newfront.append(g)
        frontier=newfront
    return found,edits

def main():
    data=json.loads(DATA.read_text(encoding='utf-8'))['datasets_by_n_observations']
    train_items=[it for key,group in data.items() if int(key)>=2 for it in group]
    test_items=[it for it in data['1'] if it['observation_set_id'] in ('n1_004','n1_006','n1_007')]
    # Learn which local graph edits preserve compatibility using train observation sets only.
    xs=[];ys=[]; train_positive_hashes=0
    selected_items=RNG.sample(train_items,min(40,len(train_items)))
    for item in selected_items:
        positives=item['ground_truth_graphs']; pos={h(g) for g in positives}
        # Sample up to 160 compatible DAGs per training task for CPU-sized fitting.
        selected=positives if len(positives)<=40 else RNG.sample(positives,40)
        for g in selected:
            train_positive_hashes+=1; xs.append(features(g,item['observations']));ys.append(1.)
            for v in variants(g,2):
                xs.append(features(v,item['observations']));ys.append(float(h(v) in pos))
    X=np.stack(xs);Y=np.asarray(ys,float)
    mean=X.mean(0);scale=X.std(0)+1e-6;Z=(X-mean)/scale;w=np.zeros(Z.shape[1]);b=0.
    for epoch in range(700):
        p=sigmoid(Z@w+b);lr=.08/(1+epoch/250)
        w-=lr*((Z.T@(p-Y))/len(Y)+.02*w);b-=lr*float(np.mean(p-Y))
    model=(mean,scale,w,b)
    mean,scale,w,b=model
    def score(g,obs):return float(sigmoid(((features(g,obs)-mean)/scale)@w+b))
    # Build held-out test records. Baseline uses the first 100 API hypotheses.
    reports=[]
    for item in test_items:
        sid=item['observation_set_id'];obs=item['observations'];gt={h(g) for g in item['ground_truth_graphs']}
        raw=json.loads((SAMPLES/f"n1_{sid}.json").read_text(encoding='utf-8'))
        calls=raw['records'][:100]
        baseline=[];baseline_all=[];seen_all=set();seen=set();valid_count=0
        for rec in calls:
            g=rec.get('graph')
            if g:
                baseline_all.append(g)
                if h(g) not in seen_all:seen_all.add(h(g))
                if rec.get('valid'):valid_count+=1
                if rec.get('valid') and h(g) not in seen:seen.add(h(g));baseline.append(g)
        baseline_hashes={h(g) for g in baseline}
        baseline_gt=baseline_hashes & gt
        # Each method has the same 100-hypothesis budget: one DeepSeek seed and 99 edits.
        seed=next((g for g in baseline if g),None)
        found,edits=search(seed,obs,score,83102+sum(map(ord,sid)))
        rng_random=random.Random(991+sum(map(ord,sid)))
        found_random,edits_random=search(seed,obs,lambda g,o:rng_random.random(),83102+sum(map(ord,sid)))
        reports.append({'id':sid,'n_ground_truths':len(gt),'baseline_calls':len(calls),
            'baseline_validity':valid_count/len(calls),'baseline_unique_all':len(seen_all),
            'baseline_novelty_rate':len(seen_all)/len(calls),
            'baseline_unique_valid':len(baseline_hashes),
            'baseline_recovered_gt':len(baseline_gt),'baseline_recovery':len(baseline_gt)/len(gt),
            'edit_seed_count':int(seed is not None),
            'learned_edit':{'proposals':edits,'unique_valid':len(found),'validity':len(found)/max(1,edits+int(seed is not None)),
                'unique_all':edits+int(seed is not None),'novelty_rate':1.0 if edits+int(seed is not None)>0 else 0.0,
                'recovered_gt':len(found & gt),'recovery':len(found & gt)/len(gt),
                'new_unique_vs_baseline':len(found-baseline_hashes),'new_gt_vs_baseline':len((found & gt)-baseline_gt),
                'augmented_recovered_gt':len(baseline_gt | (found & gt)),'augmented_recovery':len(baseline_gt | (found & gt))/len(gt)},
            'random_edit_control':{'proposals':edits_random,'unique_valid':len(found_random),'validity':len(found_random)/max(1,edits_random+int(seed is not None)),
                'unique_all':edits_random+int(seed is not None),'novelty_rate':1.0 if edits_random+int(seed is not None)>0 else 0.0,
                'recovered_gt':len(found_random & gt),'recovery':len(found_random & gt)/len(gt),
                'new_gt_vs_baseline':len((found_random & gt)-baseline_gt)},
            'model_generated_train_rows':len(Y)})
    report={'method':'learned compatibility-guided DAG edit search','train_split':'observation counts >=2, sampled 40 observation sets','test_split':'held-out nobs=1 tasks n1_004/n1_006/n1_007','negative_examples':'one-edge DAG mutations labeled by exact training-task compatibility','budget':'99 local graph edit proposals plus one DeepSeek seed (100 hypotheses); baseline first 100 cached DeepSeek calls','train_rows':len(Y),'train_positives_sampled':train_positive_hashes,'test':reports}
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
