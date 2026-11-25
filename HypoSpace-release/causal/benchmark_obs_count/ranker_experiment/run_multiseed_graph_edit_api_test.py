"""Fresh API A/B: native DeepSeek outputs vs ML-ranked graph edits from the same API pool."""
import json, os, sys, time, random
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent;CAUSAL=HERE.parents[1]
sys.path.insert(0,str(CAUSAL));sys.path.insert(0,str(HERE))
from run_causal_benchmark import CausalBenchmarkEnhanced,setup_llm
from modules.models import CausalGraph
from multi_seed_valid_unique_search import gh,dag,valid,train
DATA=CAUSAL/'benchmark_obs_count'/'n5_obs1to5_80each.json'
TARGET=os.environ.get('HYPOSPACE_TEST_TASK_ID','n1_006')
OUT=HERE/f'multiseed_graph_edit_api_{TARGET}.json'
BUDGET=100;NODES=['A','B','C','D','E'];PAIRS=[(a,b) for a in NODES for b in NODES if a!=b]
def metrics(graphs,obs,gt):
 hashes=[gh(g) for g in graphs];valid_hashes={gh(g) for g in graphs if valid(g,obs)}
 return {'final_outputs':len(graphs),'valid_outputs':sum(valid(g,obs) for g in graphs),'validity':sum(valid(g,obs) for g in graphs)/max(1,len(graphs)),
  'unique_all':len(set(hashes)),'uniqueness':len(set(hashes))/max(1,len(graphs)),
  'recovered_gt':len(valid_hashes&gt),'recovery':len(valid_hashes&gt)/len(gt)}
def save(x):
 tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(OUT)
def main():
 key=os.environ.get('HYPOSPACE_DEEPSEEK_API_KEY')
 if not key:raise RuntimeError('HYPOSPACE_DEEPSEEK_API_KEY missing from process environment')
 dataset=json.loads(DATA.read_text(encoding='utf-8'));groups=dataset['datasets_by_n_observations']
 task=next(x for x in groups['1'] if x['observation_set_id']==TARGET);obs=task['observations'];gt={gh(g) for g in task['ground_truth_graphs']}
 scorer,train_rows=train(groups);bench=CausalBenchmarkEnhanced(str(DATA));llm=setup_llm('deepseek',model='deepseek-flash',api_key=key,temperature=.7)
 result={'experiment':'fresh DeepSeek API pool; native vs ML-ranked one-edge edits','target_id':TARGET,'ground_truth_count':len(gt),'api_pool_budget':BUDGET,'final_output_budget_per_arm':BUDGET,
  'training':'logistic regression on n_observations>=2; held-out n1_006 excluded','train_rows':train_rows,'arms':{'deepseek_native':[]},'api_cost':0.0,'api_prompt_tokens':0,'api_completion_tokens':0}
 if OUT.exists():
  old=json.loads(OUT.read_text(encoding='utf-8'))
  if old.get('target_id')==TARGET and old.get('api_pool_budget')==BUDGET:result=old
 records=result['arms']['deepseek_native'];hyps=[CausalGraph.from_dict(r['graph']) for r in records if r.get('graph')]
 for q in range(len(records),BUDGET):
  data=llm.query_with_usage(bench.create_prompt(obs,hyps));response=data['response'];graph=bench.parse_llm_response(response)
  rec={'query':q+1,'response':response,'graph':graph.to_dict() if graph else None,'usage':data.get('usage',{}),'cost':data.get('cost',0.)}
  records.append(rec);result['api_cost']+=rec['cost'];result['api_prompt_tokens']+=rec['usage'].get('prompt_tokens',0);result['api_completion_tokens']+=rec['usage'].get('completion_tokens',0)
  if graph:hyps.append(graph)
  result['native_metrics']=metrics([r['graph'] for r in records if r.get('graph')],obs,gt);save(result)
  print(f"API pool {q+1}/{BUDGET}: parsed={graph is not None}; cost=${rec['cost']:.4f}",flush=True);time.sleep(.15)
 seeds=[];seen_seed=set()
 for r in records:
  g=r.get('graph')
  if g and valid(g,obs) and gh(g) not in seen_seed:seeds.append(g);seen_seed.add(gh(g))
 pool={}
 for g in seeds:
  edges=set(map(tuple,g['edges']))
  for a,b in PAIRS:
   e=set(edges)
   if (a,b) in e:e.remove((a,b))
   else:e.add((a,b))
   if dag(e):
    c={'nodes':NODES,'edges':sorted(e)};h=gh(c)
    if h not in seen_seed:pool[h]=c
 candidates=list(pool.values());ranked=sorted(candidates,key=lambda g:(-scorer(g,obs),gh(g)))[:BUDGET]
 rng=random.Random(63017);random_ranked=rng.sample(candidates,min(BUDGET,len(candidates)))
 result['candidate_pool']={'seed_count':len(seeds),'dag_edit_candidates':len(candidates),'all_unique':True}
 result['arms']['ml_ranked_edit']={'graphs':ranked,'metrics':metrics(ranked,obs,gt),'scorer':'learned logistic compatibility prior'}
 result['arms']['random_edit_control']={'graphs':random_ranked,'metrics':metrics(random_ranked,obs,gt),'scorer':'random sample from same candidate pool'}
 result['native_metrics']=metrics([r['graph'] for r in records if r.get('graph')],obs,gt)
 result['comparison']={name:{m:arm['metrics'][m]-result['native_metrics'][m] for m in ('validity','uniqueness','recovery')} for name,arm in result['arms'].items() if name!='deepseek_native'}
 result['evaluation_note']='Same fresh 100-call API pool seeds both arms; ML edits and outputs are local computations. Report API spend separately from final hypothesis count.'
 save(result);print(json.dumps({'native':result['native_metrics'],'ml':result['arms']['ml_ranked_edit']['metrics'],'random':result['arms']['random_edit_control']['metrics'],'api_cost':result['api_cost'],'candidate_pool':result['candidate_pool'],'comparison':result['comparison']},indent=2),flush=True)
if __name__=='__main__':main()
