"""Offline held-out test of an ML ranker trained to predict GT recovery."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; CAUSAL=HERE.parents[1]
sys.path.insert(0,str(HERE))
from evaluate_cached_ranker import features,fit,sigmoid,graph_hash
DATA=CAUSAL/'benchmark_obs_count'/'n5_obs1to5_80each.json'
SAMPLES=CAUSAL/'benchmark_obs_count'/'saturation'
OUT=HERE/'recovery_ranker_offline.json'
def main():
 ds=json.loads(DATA.read_text(encoding='utf-8'))['datasets_by_n_observations']
 obs={x['observation_set_id']:x['observations'] for group in ds.values() for x in group}
 train=[]
 for p in SAMPLES.glob('n*.json'):
  d=json.loads(p.read_text(encoding='utf-8'))
  if int(d['n_observations'])<2: continue
  for r in d['records']:
   if r.get('graph'): train.append((features(r['graph'],obs[d['id']]),float(r.get('recovered_gt',False))))
 X=np.stack([x for x,y in train]);Y=np.array([y for x,y in train])
 model=fit(X,Y);mean,scale,w,b=model
 def score(g,o):return float(sigmoid(((features(g,o)-mean)/scale)@w+b))
 reports=[]
 for sid in ('n1_004','n1_006','n1_007'):
  item=next(x for x in ds['1'] if x['observation_set_id']==sid); gt={graph_hash(g) for g in item['ground_truth_graphs']}
  records=json.loads((SAMPLES/f'n1_{sid}.json').read_text(encoding='utf-8'))['records'][:100]
  pool=[];seen=set()
  for r in records:
   g=r.get('graph')
   if g and graph_hash(g) not in seen: seen.add(graph_hash(g));pool.append(r)
  ranked=sorted(enumerate(pool),key=lambda z:(-score(z[1]['graph'],item['observations']),z[0]))
  k=min(50,len(pool)); chosen=[r for _,r in ranked[:k]]; baseline=pool[:k]
  def m(rows):
   valid=[r for r in rows if r.get('valid')]; recovered={graph_hash(r['graph']) for r in valid}&gt
   return {'n':len(rows),'validity':len(valid)/max(1,len(rows)),'uniqueness':len({graph_hash(r['graph']) for r in rows})/max(1,len(rows)),'recovered_gt':len(recovered),'recovery':len(recovered)/len(gt)}
  reports.append({'id':sid,'pool_candidates':len(pool),'final_budget':k,'baseline_first_k':m(baseline),'ml_top_k':m(chosen)})
 report={'method':'logistic regression trained on recovered_gt labels from nobs>=2 API outputs','train_rows':len(Y),'train_positive_rate':float(Y.mean()),'test_tasks':'held-out nobs=1; first 100 cached API candidates; rank top 50 vs first 50','results':reports}
 OUT.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
