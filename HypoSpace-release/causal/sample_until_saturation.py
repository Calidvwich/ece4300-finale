"""Stratified DeepSeek sampling until a configurable novelty plateau."""
import argparse, json, os, random, time, atexit
from pathlib import Path
import yaml
from run_causal_benchmark import CausalBenchmarkEnhanced, setup_llm
from modules.models import CausalGraph

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--per-count',type=int,default=3); ap.add_argument('--stale-stop',type=int,default=5)
    ap.add_argument('--seed',type=int,default=33550336)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    marker=out/'sampling.pid'
    marker.write_text(str(os.getpid()),encoding='ascii')
    def clear_marker():
        try:
            if marker.exists() and marker.read_text(encoding='ascii').strip()==str(os.getpid()): marker.unlink()
        except OSError: pass
    atexit.register(clear_marker)
    heartbeat=out/'sampling.heartbeat'
    def touch_heartbeat(): heartbeat.write_text(str(time.time()),encoding='ascii')
    d=json.loads(Path(a.dataset).read_text(encoding='utf-8'))
    rows=d['datasets_by_n_observations']; cfg=yaml.safe_load(Path('causal/config/config_deepseek.yaml').read_text())
    llmcfg=cfg['llm']; model=llmcfg['models']['deepseek']
    llm=setup_llm('deepseek',model=model,api_key=os.environ.get('HYPOSPACE_DEEPSEEK_API_KEY'),temperature=0.7)
    bench=CausalBenchmarkEnhanced(a.dataset)
    rng=random.Random(a.seed); summary=[]
    for nobs_s, group in rows.items():
        nobs=int(nobs_s); ordered=sorted(group,key=lambda x:x['n_compatible_graphs'])
        # Select low/median/high solution-space cases, deterministically.
        idxs=sorted(set([0,len(ordered)//2,len(ordered)-1]))[:a.per_count]
        for rank,idx in enumerate(idxs):
            item=ordered[idx]; seen=set(); stale=0; records=[]; cost=0.; pt=ct=0
            result_path=out/f'n{nobs}_{item["observation_set_id"]}.json'
            if result_path.exists():
                previous=json.loads(result_path.read_text(encoding='utf-8'))
                records=previous.get('records',[])
                seen={CausalGraph.from_dict(r['graph']).get_hash() for r in records if r.get('new_valid_unique') and r.get('graph')}
                stale=records[-1].get('stale_after',0) if records else 0
                cost=previous.get('cost',0.); pt=previous.get('prompt_tokens',0); ct=previous.get('completion_tokens',0)
                if stale >= a.stale_stop:
                    summary.append({k:v for k,v in previous.items() if k!='records'})
                    print(json.dumps(summary[-1]),flush=True)
                    continue
            gts={CausalGraph.from_dict(g).get_hash() for g in item['ground_truth_graphs']}
            while stale<a.stale_stop:
                touch_heartbeat()
                prompt=bench.create_prompt(item['observations'],[CausalGraph.from_dict(x['graph']) for x in records if x.get('graph')])
                res=llm.query_with_usage(prompt); pt+=res['usage'].get('prompt_tokens',0); ct+=res['usage'].get('completion_tokens',0); cost+=res.get('cost',0.)
                touch_heartbeat()
                graph=bench.parse_llm_response(res['response']); valid=bool(graph and bench.validate_hypothesis(graph,item['observations']))
                novel=bool(valid and graph.get_hash() not in seen)
                if novel: seen.add(graph.get_hash()); stale=0
                else: stale+=1
                records.append({'query':len(records)+1,'response':res['response'],'graph':graph.to_dict() if graph else None,'valid':valid,'new_valid_unique':novel,'recovered_gt':bool(valid and graph.get_hash() in gts),'stale_after':stale,'usage':res['usage'],'cost':res.get('cost',0.)})
                tmp={'n_observations':nobs,'id':item['observation_set_id'],'n_ground_truths':len(gts),'n_queries':len(records),'n_valid_unique':len(seen),'recovery':len(seen & gts)/len(gts),'prompt_tokens':pt,'completion_tokens':ct,'cost':cost,'records':records}
                result_path.write_text(json.dumps(tmp,indent=2),encoding='utf-8')
            summary.append({k:v for k,v in tmp.items() if k!='records'})
            print(json.dumps(summary[-1]),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()
