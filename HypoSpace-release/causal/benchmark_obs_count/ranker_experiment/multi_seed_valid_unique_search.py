"""Multi-seed learned edit search using cached real DeepSeek API outputs."""
import hashlib,json,random,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; CAUSAL=HERE.parents[1]
sys.path.insert(0,str(CAUSAL));sys.path.insert(0,str(HERE))
from evaluate_cached_ranker import features,sigmoid
DATA=CAUSAL/'benchmark_obs_count'/'n5_obs1to5_80each.json'
SAMPLES=CAUSAL/'benchmark_obs_count'/'saturation'
API=HERE/'graph_edit_api_ab_n1_006.json'
OUT=HERE/'multi_seed_valid_unique_report.json'
NODES=['A','B','C','D','E'];PAIRS=[(a,b) for a in NODES for b in NODES if a!=b]
def gh(g):return hashlib.sha256(json.dumps(sorted(map(tuple,g['edges'])),separators=(',',':')).encode()).hexdigest()
def dag(edges):
 adj={n:[] for n in NODES}
 for a,b in edges:adj[a].append(b)
 state={n:0 for n in NODES}
 def visit(n):
  if state[n]==1:return False
  if state[n]==2:return True
  state[n]=1
  for v in adj[n]:
   if not visit(v):return False
  state[n]=2;return True
 return all(visit(n) for n in NODES)
def valid(g,obs):
 edges=set(map(tuple,g['edges']))
 if not dag(edges):return False
 adj={n:[] for n in NODES}
 for a,b in edges:adj[a].append(b)
 for o in obs:
  seen=set();stack=list(adj[o['perturbed_node']])
  while stack:
   x=stack.pop()
   if x not in seen:seen.add(x);stack.extend(adj[x])
  if any(int(n in seen)!=int(o['effects'].get(n,0)) for n in NODES):return False
 return True
def train(ds):
 rng=random.Random(83102);xs=[];ys=[]
 tasks=[it for k,grp in ds.items() if int(k)>=2 for it in grp]
 for task in rng.sample(tasks,min(40,len(tasks))):
  gt={gh(g) for g in task['ground_truth_graphs']}
  for g in rng.sample(task['ground_truth_graphs'],min(40,len(task['ground_truth_graphs']))):
   xs.append(features(g,task['observations']));ys.append(1.)
   e=set(map(tuple,g['edges']));p=PAIRS[:];rng.shuffle(p);num=0
   for a,b in p:
    v=set(e)
    if (a,b) in v:v.remove((a,b))
    else:v.add((a,b))
    if dag(v):
     q={'nodes':NODES,'edges':sorted(v)};xs.append(features(q,task['observations']));ys.append(float(gh(q) in gt));num+=1
    if num==2:break
 X=np.stack(xs);Y=np.array(ys);mu=X.mean(0);sd=X.std(0)+1e-6;Z=(X-mu)/sd;w=np.zeros(Z.shape[1]);b=0.
 for ep in range(700):
  pr=sigmoid(Z@w+b);lr=.08/(1+ep/250);w-=lr*((Z.T@(pr-Y))/len(Y)+.02*w);b-=lr*float(np.mean(pr-Y))
 return lambda g,o:float(sigmoid(((features(g,o)-mu)/sd)@w+b)),len(Y)
def main():
 dataset=json.loads(DATA.read_text(encoding='utf-8'))['datasets_by_n_observations'];score,ntrain=train(dataset)
 api=json.loads(API.read_text(encoding='utf-8'));results=[]
 for sid in ['n1_004','n1_006','n1_007']:
  task=next(x for x in dataset['1'] if x['observation_set_id']==sid);obs=task['observations'];gt={gh(g) for g in task['ground_truth_graphs']}
  if sid=='n1_006':raw=api['arms']['deepseek_baseline']
  else:raw=json.loads((SAMPLES/f'n1_{sid}.json').read_text(encoding='utf-8'))['records'][:100]
  seeds=[];baseline=set()
  for r in raw:
   g=r.get('graph')
   if g and r.get('valid') and gh(g) not in baseline:baseline.add(gh(g));seeds.append(g)
  pool={}
  for g in seeds:
   e=set(map(tuple,g['edges']))
   for a,b in PAIRS:
    q=set(e)
    if (a,b) in q:q.remove((a,b))
    else:q.add((a,b))
    if dag(q):
     cand={'nodes':NODES,'edges':sorted(q)};key=gh(cand)
     if key not in baseline:pool[key]=cand
  candidates=list(pool.values())
  ml=sorted(candidates,key=lambda g:(-score(g,obs),gh(g)))[:100]
  rr=random.Random(441+sum(map(ord,sid)));rnd=rr.sample(candidates,min(100,len(candidates)))
  def metric(gs):
   hashes={gh(g) for g in gs};valid_hashes={gh(g) for g in gs if valid(g,obs)};return {'proposals':len(gs),'validity':sum(valid(g,obs) for g in gs)/max(1,len(gs)),'unique_all':len(hashes),'uniqueness':len(hashes)/max(1,len(gs)),'unique_valid':len(valid_hashes),'recovered_gt':len(valid_hashes&gt),'recovery':len(valid_hashes&gt)/len(gt)}
  results.append({'id':sid,'seed_count':len(seeds),'dag_edit_pool':len(candidates),'baseline_first100':{'validity':sum(r.get('valid',False) for r in raw[:100])/100,'uniqueness':len(baseline)/100,'recovered_gt':len(baseline&gt),'recovery':len(baseline&gt)/len(gt)},'ml_top100':metric(ml),'random100':metric(rnd)})
 report={'method':'multi-seed one-edge DAG edit proposal ranking; logistic prior trained on other observation sets','train_rows':ntrain,'selection':'same pool of DAG edits excluding seed graphs; compare top 100 by learned score vs random order, validator labels proposals','results':results}
 OUT.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
