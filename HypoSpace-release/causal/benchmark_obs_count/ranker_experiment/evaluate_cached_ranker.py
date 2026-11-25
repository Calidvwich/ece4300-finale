"""Offline grouped evaluation for the learned DeepSeek candidate reranker."""
import json, math, random, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark_obs_count" / "n5_obs1to5_80each.json"
SAMPLES = ROOT / "benchmark_obs_count" / "saturation"
OUT = ROOT / "benchmark_obs_count" / "ranker_experiment"
NODES = ["A", "B", "C", "D", "E"]
EDGES = [(a,b) for a in range(5) for b in range(5) if a != b]

def graph_hash(g):
    return hashlib.sha256(json.dumps(sorted(map(tuple,g['edges'])),separators=(',',':')).encode()).hexdigest()

def features(g, obs):
    ix={n:i for i,n in enumerate(NODES)}
    adj=np.zeros((5,5),dtype=float)
    for a,b in g.get('edges',[]):
        if a in ix and b in ix: adj[ix[a],ix[b]]=1
    f=list(adj.ravel())
    indeg=adj.sum(0); outdeg=adj.sum(1)
    f.extend(indeg); f.extend(outdeg); f.extend([adj.sum(),np.trace(adj@adj),np.trace(adj@adj@adj)])
    # Observation-conditioned reachability residual features.
    closure=adj.copy()
    for k in range(5):
        for i in range(5):
            for j in range(5): closure[i,j]=max(closure[i,j],closure[i,k]*closure[k,j])
    for slot in range(5):
        if slot >= len(obs):
            f.extend([0.0]*11)
            continue
        o=obs[slot]
        p=ix[o['perturbed_node']]
        expected=np.array([o['effects'].get(n,0) for n in NODES],float)
        predicted=closure[p]
        f.extend(predicted.tolist())
        f.extend(np.abs(predicted-expected).tolist())
        f.append(float(np.sum(predicted != expected)))
    return np.asarray(f,dtype=float)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def fit(x,y,seed=51):
    mean=x.mean(0); scale=x.std(0)+1e-6; z=(x-mean)/scale
    w=np.zeros(z.shape[1]); b=0.; rng=np.random.default_rng(seed)
    for epoch in range(2500):
        p=sigmoid(z@w+b); lr=.1/(1+epoch/700)
        w-=lr*((z.T@(p-y))/len(y)+.015*w); b-=lr*float(np.mean(p-y))
    return mean,scale,w,b

def main():
    dataset=json.loads(DATA.read_text(encoding='utf-8'))['datasets_by_n_observations']
    obs_by_id={r['observation_set_id']:r['observations'] for group in dataset.values() for r in group}
    examples=[]
    for path in sorted(SAMPLES.glob('n*.json')):
        d=json.loads(path.read_text(encoding='utf-8')); obs=obs_by_id[d['id']]
        for rec in d['records']:
            if rec.get('graph'):
                examples.append({'id':d['id'],'nobs':d['n_observations'],'hash':graph_hash(rec['graph']),
                    'x':features(rec['graph'],obs),'y':float(rec['valid']),'recovered':bool(rec['recovered_gt'])})
    # Hold out low-observation, large-hypothesis-space cases (the hard slice).
    test=[e for e in examples if e['nobs']==1]
    # Candidate identity may overlap across observations: observation-conditioned
    # features and disjoint observation-set ids keep the prediction examples distinct.
    train=[e for e in examples if e['nobs']>=2]
    if not train or len({e['y'] for e in train})<2: raise RuntimeError('Insufficient grouped training classes')
    x=np.stack([e['x'] for e in train]); y=np.array([e['y'] for e in train]); model=fit(x,y)
    def score(e):
        mean,scale,w,b=model
        return float(sigmoid(((e['x']-mean)/scale)@w+b))
    groups={}
    for e in test: groups.setdefault(e['id'],[]).append(e)
    rows=[]
    for sid,items in sorted(groups.items()):
        # Stable query-order baseline; candidate deduplication matches benchmark novelty.
        seen=set(); pool=[]
        for e in items:
            if e['hash'] not in seen: seen.add(e['hash']); pool.append(e)
        ranked=sorted(enumerate(pool),key=lambda z:(-score(z[1]),z[0]))
        row={'id':sid,'nobs':items[0]['nobs'],'candidates':len(pool),'baseline':{},'ml':{}}
        for k in (1,5,10,20):
            bpool=pool[:k]; mpool=[e for _,e in ranked[:k]]
            row['baseline'][str(k)]={'precision':sum(e['y'] for e in bpool)/max(1,len(bpool)),'valid_found':sum(e['y'] for e in bpool)}
            row['ml'][str(k)]={'precision':sum(e['y'] for e in mpool)/max(1,len(mpool)),'valid_found':sum(e['y'] for e in mpool)}
        pos=[i for i,e in enumerate(pool) if e['y']]; posrank=next((r+1 for r,(_,e) in enumerate(ranked) if e['y']),None)
        row['mrr_baseline']=1/(pos[0]+1) if pos else 0.; row['mrr_ml']=1/posrank if posrank else 0.
        rows.append(row)
    metrics={}
    for mode in ('baseline','ml'):
        metrics[mode]={}
        for k in ('1','5','10','20'):
            vals=[r[mode][k]['precision'] for r in rows]
            metrics[mode][f'precision@{k}']=float(np.mean(vals))
        metrics[mode]['MRR']=float(np.mean([r['mrr_'+mode] for r in rows]))
    report={'method':'numpy_logistic_validity_reranker','target':'validity on hard low-observation slice','split':'train n_observations>=2; test n_observations=1; observation-set ids disjoint; candidate graphs may recur under different observations','train_candidates':len(train),'test_candidates':len(test),'test_instances':len(rows),'metrics':metrics,'per_instance':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'offline_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('method','split','train_candidates','test_candidates','test_instances','metrics')},indent=2))

if __name__=='__main__': main()
