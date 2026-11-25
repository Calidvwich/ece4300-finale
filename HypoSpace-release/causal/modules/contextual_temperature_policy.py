"""LinUCB temperature controller warm-started from prior benchmark traces."""
import json
from pathlib import Path
import numpy as np

ARMS=(0.3,0.7,1.0)
DIM=6

def make_context(query_index, recent_rewards, stale_streak, n_observations, n_ground_truths, budget):
    recent=list(recent_rewards)[-5:]
    mean_recent=float(np.mean(recent)) if recent else 0.0
    return np.asarray([
        1.0,
        query_index/max(1,budget-1),
        mean_recent,
        min(stale_streak,5)/5.0,
        n_observations/5.0,
        min(np.log1p(n_ground_truths)/np.log1p(10000),1.0),
    ],dtype=float)

class LinUCBTemperaturePolicy:
    def __init__(self, alpha=0.2, ridge=1.0):
        self.alpha=alpha
        self.A={a:np.eye(DIM)*ridge for a in ARMS}
        self.b={a:np.zeros(DIM) for a in ARMS}

    def update(self, arm, context, reward):
        x=np.asarray(context,float)
        self.A[arm]+=np.outer(x,x)
        self.b[arm]+=x*float(reward)

    def choose(self, context):
        x=np.asarray(context,float); rows=[]
        for arm in ARMS:
            inv=np.linalg.inv(self.A[arm]); theta=inv@self.b[arm]
            mean=float(x@theta); uncertainty=float(np.sqrt(max(0.0,x@inv@x)))
            rows.append((mean+self.alpha*uncertainty,arm,mean,uncertainty))
        best=max(rows,key=lambda z:(z[0],-ARMS.index(z[1])))
        return best[1],{'ucb':best[0],'predicted_reward':best[2],'uncertainty':best[3],
                        'arm_scores':{str(z[1]):{'ucb':z[0],'mean':z[2]} for z in rows}}

    def warm_start(self, trace_path):
        data=json.loads(Path(trace_path).read_text(encoding='utf-8'))
        n=0
        for sample in data.get('per_sample_results',[]):
            trace=sample.get('temperature_trace',[]); rewards=[]; stale=0
            nobs=sample.get('n_observations',5); ngt=sample.get('n_ground_truths',1)
            budget=max(1,len(trace))
            for z in trace:
                arm=float(z['temperature'])
                if arm not in ARMS: continue
                ctx=make_context(z.get('query',len(rewards)),rewards,stale,nobs,ngt,budget)
                reward=float(z.get('marginal_valid_unique',0.0))
                self.update(arm,ctx,reward); rewards.append(reward)
                stale=0 if reward>0 else stale+1; n+=1
        return n
