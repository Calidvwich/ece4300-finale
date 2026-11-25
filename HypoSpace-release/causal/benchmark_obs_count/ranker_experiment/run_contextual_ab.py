"""Paired fixed-temperature vs online LinUCB DeepSeek API experiment."""
import json, os, random, sys, time
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
CAUSAL=HERE.parents[1]
ROOT=CAUSAL.parent
sys.path.insert(0,str(CAUSAL))
sys.path.insert(0,str(CAUSAL/'modules'))
from run_causal_benchmark import CausalBenchmarkEnhanced, setup_llm
from modules.models import CausalGraph
from modules.contextual_temperature_policy import LinUCBTemperaturePolicy,make_context

DATA=CAUSAL/'benchmark_obs_count'/'n5_obs1to5_80each.json'
WARM=CAUSAL/'benchmark_temperature_ucb_chatapi'/'deepseek_n5_ucb.json'
OUT=CAUSAL/'benchmark_obs_count'/'ranker_experiment'/'contextual_ab_n1_004_100.json'
TARGET_ID='n1_004'  # one-observation, 960 compatible DAGs
BUDGET=100

def main():
    cfg=yaml.safe_load((CAUSAL/'config'/'config_deepseek.yaml').read_text(encoding='utf-8'))
    key=os.environ.get('HYPOSPACE_DEEPSEEK_API_KEY')
    if not key: raise RuntimeError('HYPOSPACE_DEEPSEEK_API_KEY missing')
    llm=setup_llm('deepseek',model=cfg['llm']['models']['deepseek'],api_key=key,temperature=.7)
    bench=CausalBenchmarkEnhanced(str(DATA))
    raw=json.loads(DATA.read_text(encoding='utf-8'))['datasets_by_n_observations']
    item=next(x for grp in raw.values() for x in grp if x['observation_set_id']==TARGET_ID)
    obs=item['observations']; gt={CausalGraph.from_dict(g).get_hash() for g in item['ground_truth_graphs']}
    # Randomize arm order to reduce order/fatigue drift; independent history per arm.
    order=['fixed_0.7','linucb']
    policy=LinUCBTemperaturePolicy(alpha=.20); warm_n=policy.warm_start(WARM)
    result={'dataset_id':TARGET_ID,'n_observations':len(obs),'n_ground_truths':len(gt),'budget_per_arm':BUDGET,
            'arm_order':order,'policy':'LinUCB context=[intercept, progress, recent reward, stale streak, obs count, log GT-space]; warm-started from prior UCB trajectory','warm_start_steps':warm_n,'arms':{}}
    if OUT.exists():
        saved=json.loads(OUT.read_text(encoding='utf-8'))
        if saved.get('dataset_id')==TARGET_ID and saved.get('budget_per_arm')==BUDGET:
            result=saved
    for arm_name in order:
        if arm_name in result['arms'] and result['arms'][arm_name].get('n_queries',0)>=BUDGET:
            continue
        llm.temperature=.7; hypotheses=[]; seen=set(); records=[]; rewards=[]; stale=0
        total_cost=0.; prompt_tokens=completion_tokens=0; recovered=set()
        for q in range(BUDGET):
            if arm_name=='linucb':
                context=make_context(q,rewards,stale,len(obs),len(gt),BUDGET)
                temperature,decision=policy.choose(context)
            else:
                context=make_context(q,rewards,stale,len(obs),len(gt),BUDGET)
                temperature=.7; decision={'control':'fixed'}
            llm.temperature=temperature
            prompt=bench.create_prompt(obs,hypotheses)
            response_data=llm.query_with_usage(prompt)
            response=response_data['response']; total_cost+=response_data.get('cost',0.)
            usage=response_data.get('usage',{}); prompt_tokens+=usage.get('prompt_tokens',0); completion_tokens+=usage.get('completion_tokens',0)
            graph=bench.parse_llm_response(response)
            valid=bool(graph and bench.validate_hypothesis(graph,obs))
            h=graph.get_hash() if graph else None
            novel=bool(valid and h not in seen)
            reward=float(novel)
            if graph: hypotheses.append(graph)
            if novel:
                seen.add(h); stale=0
                if h in gt: recovered.add(h)
            else: stale+=1
            rewards.append(reward)
            if arm_name=='linucb': policy.update(temperature,context,reward)
            records.append({'query':q+1,'temperature':temperature,'context':context.tolist(),'decision':decision,
                'response':response,'graph':graph.to_dict() if graph else None,'valid':valid,'new_valid_unique':novel,
                'recovered_gt':bool(valid and h in gt),'reward':reward,'usage':usage,'cost':response_data.get('cost',0.)})
            partial={'n_queries':q+1,'valid':sum(r['valid'] for r in records),'unique_valid':len(seen),
                'recovered_ground_truths':len(recovered),'recovery':len(recovered)/len(gt),'cost':total_cost,
                'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,'records':records}
            result['arms'][arm_name]=partial
            OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'arm':arm_name,**{k:v for k,v in result['arms'][arm_name].items() if k!='records'}}),flush=True)
    fixed=result['arms']['fixed_0.7']; ml=result['arms']['linucb']
    result['comparison']={'recovery_delta':ml['recovery']-fixed['recovery'],
       'valid_delta':ml['valid']/BUDGET-fixed['valid']/BUDGET,'unique_delta':ml['unique_valid']-fixed['unique_valid'],
       'cost_delta':ml['cost']-fixed['cost']}
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['comparison']),flush=True)

if __name__=='__main__': main()
