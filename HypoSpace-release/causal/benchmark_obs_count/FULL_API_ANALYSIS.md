# HypoSpace DeepSeek API run record

## Status

All API benchmark processes for the runs listed below have exited. The two causal passes are complete; no additional API pass was started after the request to stop. The machine-generated status record is [FULL_API_STATUS.json](FULL_API_STATUS.json).

The full per-query and per-task results are preserved in the result JSON files and per-shard `result.json` files. Successful shards remove their transient runner checkpoints on completion, so the merged result JSON is the durable checkpoint for completed runs. A 10-task checkpoint from the first causal pass is also retained under `checkpoints/full_api_causal_n4_obs3/`.

## Dataset and protocol

The complete local causal set used here is `causal/datasets/n4_obs3_all.json`: four labeled nodes, three interventions per task, 392 observation sets, and 2,172 total admissible hypotheses across tasks. For each task, the query budget was exactly `N = |H_O|`; every output file contains 392 distinct task IDs and 2,172 completed proposals, with zero recorded API errors.

The paper's general protocol describes drawing `N=|H_O|` samples from a fixed prompt and averaging metrics across instances. Its causal-domain formulation also writes generation as conditioned on the earlier-proposal history. To make that protocol difference visible, two separate runs were completed:

1. **Fixed prompt:** the new `--independent-samples` option sends the same observation prompt for every query. This follows the general fixed-prompt sampling description.
2. **Clone default:** the released runner's behavior sends prior proposals for that task in later prompts. This follows the causal-domain history-conditioned formulation and the checked-out code's default.

The fixed-prompt run is [full_repo_causal_n4_obs3_deepseek.json](full_repo_causal_n4_obs3_deepseek.json). The clone-default run is [full_repo_causal_n4_obs3_deepseek_history.json](full_repo_causal_n4_obs3_deepseek_history.json). They are different sampling protocols and must not be pooled as if they were one condition.

## Results

Metrics below are macro averages across the 392 observation sets, matching the paper's instance-averaging convention. “Pooled” rates instead combine all proposals/hypotheses before calculating the ratio; because `|H_O|` varies by task, the pooled and macro values differ substantially.

| Causal protocol | Validity (macro / pooled) | Uniqueness (macro / pooled) | Recovery (macro / pooled) | API errors | Calls |
|---|---:|---:|---:|---:|---:|
| Fixed prompt | 99.74% / 99.95% | 45.30% / 24.17% | 45.05% / 24.13% | 0 | 2,172 |
| Clone default with history | 100.00% / 100.00% | 99.28% / 98.57% | 99.28% / 98.57% | 0 | 2,172 |

Across matched tasks, adding the clone's prior-output context raised the macro Uniqueness score on 312 tasks and Recovery on 313 tasks; neither metric fell on any task. This is a descriptive protocol comparison, not evidence that the ML module improved performance. The history prompt explicitly says not to repeat prior graphs, so it is a prompt-conditioned baseline behavior and cannot be presented as the project's ML contribution under the course rule against prompt engineering.

With a fixed prompt, performance shows the underdetermination effect clearly: mean Recovery falls from 98.68% on one-ground-truth tasks to 10.42% on tasks with 16 admissible graphs. Validity stays near 100%. Thus the weak metric is coverage/diversity, not basic consistency. The clone-default history context nearly saturates this small four-node set, which also makes it a poor setting for demonstrating an ML add-on's headroom.

The two earlier secondary-domain runs are also preserved:

| Domain run | Scope in the clone | Tasks | Calls | Validity | Uniqueness | Recovery | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| 3D quickstart | grid 3, height 3, one top-view observation, one occupied position | 9 | 27 | 100% | 100% | 100% | 0 |
| Boolean basic quickstart | basic operators, depth 2 | 35 | 150 | 100% | 100% | 100% | 0 |

These are complete runs for those generated files, but they do **not** constitute the paper's entire difficulty sweep. The paper also reports 3D with `tp=2,3` views and Boolean `extended/full` settings. The cloned 3D generator/runner only accepts a single top view. Local extended/full Boolean datasets and a seven-task five-node `|H_O|=160` dataset were generated for possible continuation, but no API calls were made on them after the stop request. No six-node causal dataset exists in this clone.

## Cost accounting caution

The `cost` fields in the result JSON are not DeepSeek invoices. In this clone, `OpenAILLM.get_model_pricing()` does not contain `deepseek-flash` and therefore charges its fallback OpenAI rates of `$10/$30` per million input/output tokens. That is why the result JSON shows `$63.08` for the fixed-prompt causal pass and `$106.37` for the history-conditioned pass.

At the current official DeepSeek off-peak rate card, `deepseek-flash` is `$0.003` per million cached input tokens, `$0.15` per million uncached input tokens, and `$0.60` per million output tokens. Using the saved input/output token totals gives a rate-card range of about `$1.12–$1.23` for the fixed-prompt causal pass and `$1.96–$2.08` for the history-conditioned pass, depending on cache hits. Exact actual billing cannot be reconstructed because the runner did not save cached-token counts. The provider's usage page remains authoritative.

## Statistical interpretation and limitations

- These are one stochastic run per task and protocol, not multiple independent repetitions per task. The paired task comparison is useful diagnostically but does not establish an ML treatment effect.
- The runner's `p_value` is a one-sample test against zero, not a comparison between an ML method and baseline. Constant ceiling scores also caused SciPy precision warnings. Those values should not be cited as proof of significance.
- The present causal API sweep covers the available four-node/three-intervention collection only. It does not reproduce the paper's complete 4/5/6-node table or the added large-space `|H_O|=160` experiment across node counts.
- The 3D and Boolean metrics are at ceiling on the quickstart files, so those files cannot show measurable improvement. Continue on harder supported data, or implement the missing 3D multi-view observation support, before claiming whole-paper coverage.
- The current ML add-on has not been evaluated against these full API runs. Any next comparison needs identical task IDs, model, prompts, temperature, query budget, and decoding protocol for baseline and ML condition; the change itself must be a learned/sampling module rather than prompt wording.

## Reproducibility files

- [Fixed-prompt causal runner](run_full_causal_parallel.py)
- [Clone-default history runner](run_full_causal_parallel_history.py)
- [Causal runner change adding explicit independent mode](../run_causal_benchmark.py)
- [Local n4 causal set](../datasets/n4_obs3_all.json)
- [API run manifest and completion record](FULL_API_STATUS.json)

## References

- Paper: [PMLR proceedings page](https://proceedings.mlr.press/v306/chen26bz.html), [PDF](https://arxiv.org/pdf/2510.15614). The paper defines Validity, Uniqueness, Recovery, usually sets `N=|H_O|`, averages by task/difficulty, and gives a causal generation formulation using prior-output history.
- Current provider rates: [DeepSeek API Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/).
