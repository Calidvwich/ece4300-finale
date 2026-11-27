# EEC 4300 Final Project — Verifier-Guided Two-Stage Fine-Tuning for Set-Valued Hypothesis Sampling on HypoSpace: What Helps and What Doesn't

> **Language**: **English** | [简体中文](README_ZH.md)

This repository contains our **EEC 4300** group project on **HypoSpace** (*ICML 2026*, [PMLR Vol. 306](https://proceedings.mlr.press/v306/chen26bz.html)), which evaluates LLMs as samplers over **underdetermined scientific hypothesis spaces** under three metrics: **Validity**, **Uniqueness**, and **Recovery**.

### Key Contributions & Deliverables
- **Zero-Isomorphism-Leakage Evaluation Suite (`Causal`, `Boolean`, `3D`)**: Constructed equivalence-class splits ($S_4$ node permutation, $x\text{-}y$ variable symmetry, and $D_4$ dihedral symmetry) verified with zero train/val/test class overlap (`29` passing unit tests across the three domains).
- **Single-GPU Two-Stage Fine-Tuning Pipeline (`Qwen2.5-7B-Instruct`)**: Built epoch-rotating balanced `SFT-LoRA` + `Diversity-GRPO-LoRA` (`dual LoRA adapters`, `Zero-Dropout`, unified `T = 0.9, top_p = 0.95`, and `Step-0` validation guarding) engineered to run on a single 32 GB RTX 5090.
- **Controlled Multi-Domain Ablations**: Conducted full 4-way ID and OOD evaluations (`Base`, `SFT`, `Validity-Only GRPO`, `Diversity-GRPO`) across Causal (4-node ID + 5-node OOD), Boolean (extended ID + Full-NOR OOD), and 3D Voxel Reconstruction (1–3 block ID + 4-block OOD).
- **Rigorous Statistical Attribution & Visualization**: Implemented exact i.i.d. uniform ceiling normalization, per-task `Recovery = Valid × (unique_valid / valid)` decomposition, 10,000-resample cluster bootstrap 95% CIs, exact binomial sign tests, and 12 publication-ready 300 DPI figures.

> **Primary Metrics & Statistical Scope**: Throughout this repository, **Task Macro Recovery** and **Structure-Group Recovery** are our primary evaluation metrics; Validity, Uniqueness, ceiling-normalized Recovery (`Norm Rec`), and Pooled Recovery are secondary/diagnostic metrics. No multiple-comparison correction is applied. When the number of structure groups is small (`n <= 7`: Causal ID `5` groups, 3D ID `7` groups), cluster bootstrap is known to underestimate variance (narrow CIs) and exact sign tests have low power, so we treat those group comparisons as descriptive/directional and reserve "statistically significant" for results supported by both exact two-sided binomial sign tests ($p < 0.05$) and 95% cluster bootstrap CIs.

---

## 1. Problem, Motivation & Main Findings (Ranked by Evidence Strength)

Under HypoSpace's formal **Fixed-Prompt** independent sampling protocol (`--independent-samples`, $N = |H_O|$ queries per task without appending prior proposals into the prompt), even strong closed-source LLMs exhibit severe **mode collapse**: on our unseen-structure 4-node causal split, the recorded **`DeepSeek-Flash`** API baseline (`T = 0.7`) achieves **100.00% Validity**, yet its **Recovery drops to 38.34%** (and **18.52%** when $|H_O| \ge 9$). While conditioning prompts on prior outputs avoids duplicates at small budgets, prompt length grows linearly and total token cost grows quadratically ($O(N^2)$) under strictly serial execution; internalizing set coverage into model weights keeps inference linear ($O(N)$) and fully parallel.

We evaluate a **verifier-guided Two-Stage SFT-LoRA + Diversity-GRPO-LoRA** pipeline on `Qwen2.5-7B-Instruct` (`T = 0.9, top_p = 0.95`) under **zero-isomorphism-leakage splits** ($S_4$ / $D_4$ / $x\text{-}y$ symmetry) across all three HypoSpace domains (**Causal**, **Boolean**, **3D**). Ordered by empirical strength, our results show **what helps and what does not**:

1. **(Strong) Balanced SFT-LoRA over exhaustive GTs mitigates mode collapse relative to zero-shot baselines**: Under zero-isomorphism-leakage splits, Stage 1 SFT drives most of the Recovery gain over zero-shot `Base 7B` across all three domains and lifts effective valid-sample deduplication (`unique_valid / valid`) on Causal ID from `38.34%` (`DeepSeek-Flash`, `T = 0.7`) to `67.74%`. Comparisons against `DeepSeek-Flash` are reference-only due to temperature differences (`0.7` vs `0.9`) and access to solver-enumerated training GTs.
2. **(Moderate / Mixed) Stage 2 Diversity-GRPO reliably improves Validity when SFT is unsaturated, with small and statistically fragile additional Recovery gains**: Stage 2 significantly raises Validity on Causal ID (`+5.20%`, Group CI `[+0.42%, +8.73%]`), Boolean ID (`+23.27%`, $p = 0.0005$), and Boolean OOD (`+14.50%`, $p < 10^{-4}$). On Causal ID, per-task decomposition shows Stage 2's `+2.75%` Recovery gain over SFT comes from higher Validity (`86.71% → 91.91%`), while effective valid-sample deduplication slightly declines (`67.74% → 65.37%`). Additional Recovery over SFT is non-significant on Causal ID (`p = 0.625`, CI crosses `0`), significant on Causal 5-node OOD on a single run/seed (`+12.29%` Group, $p = 0.0129$, though sensitive to single-task groups), and mixed on Boolean (`+8.26%` ID with $p = 0.5488$, `+0.55%` OOD n.s.).
3. **(Boundary / Negative Findings) Ceiling saturation on 3D and mixed Uniqueness effects on Boolean**: When Stage 1 SFT already nears the i.i.d. uniform optimum (3D ID: `96.91%` of the exact ceiling), Stage 2 yields zero additional gain (`best_step = 0`). On Boolean, `Validity-Only GRPO` also improves ID Recovery (`22.90% → 27.89%`), `Diversity-GRPO` does not significantly outperform `Validity-Only` in Recovery, and `Diversity-GRPO` significantly reduces overall `Uniqueness` compared to SFT (`-8.94%` ID, `-20.22%` OOD).

---

## 2. Empirical Results (Zero-Isomorphism-Leakage Protocol)

### 2.1 Zero-Isomorphism-Leakage Splits & Exact i.i.d. Recovery Ceiling

All 392 tasks in `n4_obs3_all.json` collapse to **22 node-permutation ($S_4$) isomorphism equivalence classes** under canonical observation signatures (`canonical_isomorphism_key` in [data_utils.py](HypoSpace-release/causal/grpo_lora/data_utils.py)). We split **entire equivalence classes** into:
- **Train**: `13` groups (`256` `n=4` + `19` `n=3` = `275` tasks)
- **Val**: `4` groups (`60` tasks; `48` stratified tasks = `12/group` used for balanced validation)
- **Unseen-Structure ID Test**: `5` groups (`76` tasks, `400` queries)

Under Fixed-Prompt sampling with $K = |H_O|$ independent draws, expected Recovery is $\sum_{h \in H_O} [1 - (1 - p_h)^K]$. By concavity of $1 - (1 - p)^K$, **uniform sampling over $H_O$ ($p_h = 1/K$) attains the exact maximum expected Recovery** under any i.i.d. fixed-prompt distribution: $1 - (1 - 1/K)^K$ (`70.38%` macro across the 76 ID tasks; `81.41%` across the 80 OOD 5-node tasks). Single-task or small-group normalized values above `100%` (e.g., OOD `G15` at `133.33%`) reflect **finite-sample variance**, not surpassing the theoretical expectation limit. We report both raw metrics and ceiling-normalized Recovery (`Norm Rec`) alongside cluster bootstrap 95% CIs (10,000 resamples) via [group_stats.py](results_analysis/group_stats.py).

### 2.2 Causal 4-Node Unseen-Structure ID Test (76 Tasks / 5 Groups)

Each Qwen model is **trained once** (`T = 0.9, top_p = 0.95`, using `build_fixed_prompt()` matching official `create_prompt(..., prior_hypotheses=[])` inside the `Qwen2.5-7B-Instruct` chat template) and averaged over **3 evaluation sampling seeds** (`33550336, 1, 2`). `DeepSeek-Flash` is sliced from the full 392-task Fixed-Prompt run (`T = 0.7`, [config_deepseek.yaml](HypoSpace-release/causal/config/config_deepseek.yaml)).

| Model / Method | Eval Protocol | Parse | Valid | **Task Macro Rec (= Uniq)** | Task Norm Rec | **Group Rec (mean ± std)** | Group Norm Rec | Pooled Rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B Base` | Zero-shot, 3 eval seeds, `T=0.9` | 100.00% | 29.47% | 11.29% | 14.97% | 18.34% ± 22.45% | 21.26% ± 20.89% | 8.42% |
| `DeepSeek-Flash` *(ref only)* | API, 1 run, **`T=0.7`** *(non-matched temp)* | **100.00%** | **100.00%** | 38.34% | 52.47% | 47.80% ± 33.61% | 59.40% ± 31.47% | 27.00% |
| `SFT-LoRA` (`= Validity-GRPO best`)* | 3 eval seeds, `T=0.9` | 97.69% | 86.71% | 55.63% | 78.70% | 61.81% ± 17.15% | 81.26% ± 6.98% | 52.33% |
| **`Diversity-GRPO-LoRA`** | 3 eval seeds, `T=0.9` | 98.42% | **91.91%** | **58.38%** | **82.50%** | **64.37% ± 16.88%** | **84.78% ± 8.10%** | **53.50%** |

*\*Notes: (1) On Causal & 3D, $N = |H_O|$ and `ground_truth` is exhaustive, so `Uniqueness == Recovery`. (2) `Validity-Only GRPO` triggered `best_step == 0` protection (`val group recovery` dropped `55.82% → 52.17%` while `val valid` rose `77.52% → 96.27%` on the 4-group validation set — a qualitative observation of collapse, not test-evaluated independently); its retained test weights are identical to `SFT-LoRA`, so `diversity vs sft` and `diversity vs validity` are the same single comparison.*

- **Per-Task `Recovery = Valid × (unique_valid / valid)` Decomposition (Causal ID)**:

| Model | Valid Rate | **Task Macro Recovery** | Mean Effective Dedup (`unique_valid / valid`) | Group Effective Dedup |
|---|---:|---:|---:|---:|
| `DeepSeek-Flash` (`T=0.7`) | **100.00%** | 38.34% | 38.34% (`n=76`) | 47.80% |
| `SFT-LoRA` (`T=0.9`) | 86.71% | 55.63% | **67.74%** (`n=76`) | **73.74%** |
| `Diversity-GRPO-LoRA` (`T=0.9`) | 91.91% | **58.38%** | 65.37% (`n=76`) | 72.02% |

  *Key diagnostic*: On Causal ID, **Stage 1 SFT accounts for the mode-collapse reduction (`38.34% → 67.74%`)**. Stage 2's `+2.75%` Recovery delta over SFT comes **entirely from higher Validity (`86.71% → 91.91%`)**, while effective deduplication slightly decreases (`67.74% → 65.37%`).

- **Statistical Comparisons ([id_v2_stats.md](results_analysis/id_v2_stats.md), `n = 5` groups — descriptive only)**:
  - **`Diversity-GRPO` vs `SFT-LoRA`**: Task Rec **`+2.75%`**, Group Rec **`+2.56%`** (`3/1/1`, exact sign test **$p = 0.6250$**, 95% CI **`[-0.31%, +5.43%]`** crossing `0` — **not statistically significant**, though directionally positive on all 3 eval seeds: `57.75% vs 54.20%`, `58.13% vs 54.33%`, `59.27% vs 58.35%`). Only Validity improves (`+5.20%` Task / `+4.46%` Group, CI `[+0.42%, +8.73%]`, sign test $p = 0.1250$).
  - **`Diversity-GRPO` vs `DeepSeek-Flash` *(non-matched temperature reference)***: Task Rec **`+20.04%`** (`57/11/8`), Group Rec **`+16.57%`** (`4/0/1`, sign test **$p = 0.3750$**, CI `[+2.71%, +29.52%]`; note `sft - deepseek` Group CI **`[-0.19%, +28.19%]`** crosses `0`), with a clear **Validity trade-off (`-8.09%`, losing `5/5` groups)**. Because Qwen is fine-tuned on solver-enumerated exhaustive GTs while `DeepSeek-Flash` is evaluated zero-shot at `T = 0.7`, this reflects task-specific adaptation rather than superiority over frontier models.

#### Recovery Breakdown by Hypothesis Space Size $|H_O|$

| Bucket (`\|H_O\|`) | Tasks (Groups) | Exact Ceiling | `Base 7B` | `DeepSeek-Flash` (`T=0.7`) | `SFT-LoRA` | `Diversity-GRPO` | Norm Rec (`Diversity`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 4 (1) | 100.00% | 58.33% | **100.00%** | 91.67% | 91.67% | 91.67% |
| `2–4` | 36 (2) | 71.91% | 10.03% | 51.85% | 57.10% | **63.43%** | **88.22%** |
| `5–8` | 24 (1) | 65.64% | 8.51% | 17.71% | **47.92%** | 46.53% | 70.88% |
| `9+` | 12 (1) | 65.36% | 4.94% | 18.52% | 54.63% | **55.86%** | **85.48%** |

<p align="center">
  <img src="results_analysis/figures/fig1_id_ablation.png" width="49%" alt="Unseen-Structure ID Ablation"/>
  <img src="results_analysis/figures/fig2_difficulty_scaling.png" width="49%" alt="Difficulty Scaling vs Ceiling"/>
</p>

### 2.3 Causal 5-Node OOD Transfer (80 Tasks / 22 Structure Groups)

Evaluated zero-shot with a **single evaluation seed (`33550336`, `T = 0.9, top_p = 0.95`)** on [n5_obs5_hybrid80.json](HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json) (80 tasks, 22 structure groups, exact ceiling `81.41%`) and [causal_n5_gt160.json](HypoSpace-release/causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json) (7 tasks, $|H_O|=160$):

| Dataset | Model | Valid | **Task Macro Rec** | **Group Rec (mean ± std)** | Group Norm Rec | Pooled Rec | `Bucket 9+` Rec |
|---|---|---:|---:|---:|---:|---:|---:|
| `n5_obs5_hybrid80` (`80` tasks) | `Qwen2.5-7B Base` | 28.14% | 26.58% | 17.78% ± 33.72% | 18.71% ± 33.90% | 1.69% | 0.07% |
| `n5_obs5_hybrid80` (`80` tasks) | `SFT-LoRA` (`= Validity*`) | 67.54% | 56.97% | 50.92% ± 28.47% | 61.69% ± 26.17% | 28.28% | 26.43% |
| `n5_obs5_hybrid80` (`80` tasks) | **`Diversity-GRPO-LoRA`** | **70.33%** | **61.70%** | **63.21% ± 26.63%** | **78.17% ± 24.67%** | **31.38%** | **29.28%** |
| `causal_n5_gt160` (`7` tasks, `\|H_O\|=160`) | `Diversity-GRPO-LoRA` *(no baseline; stress ref)* | 19.29% | 10.71% | — | — | 10.71% | 10.71% |

- **22-group OOD statistical summary & sensitivity checks ([ood_n5_v2_stats.md](results_analysis/ood_n5_v2_stats.md))**:
  - `Diversity-GRPO` vs `SFT-LoRA`: **`+4.73%`** Task Macro Recovery, **`+3.10%`** Pooled Recovery, and **`+12.29% ± 24.88%`** Group Recovery (**12 wins / 8 ties / 2 losses**, exact sign test **$p = 0.0129$**, Cluster Bootstrap 95% CI **`[+3.50%, +23.68%]`**); `Task Dedup` increases from `82.95%` (`n=74`) to `85.03%` (`n=75`). However, **several caveats temper this single-run, single-seed result**:
    1. **Sensitivity to single-task groups**: `10` of the `22` OOD groups contain only `1` task (`n_tasks == 1`, contributing `+19.38% ± 32.60%`, `5/5/0`, $p = 0.0625$, e.g., single-task `G15` `0% → 100%` and `G09` `25% → 75%`). When restricting to multi-task groups (`n_tasks >= 2`, `12` groups), `Diversity - SFT` Group Delta drops to **`+6.39% ± 15.10%`** (`7/3/2`, $p = 0.1797$, 95% CI **`[-1.67%, +14.79%]`**, crossing `0`). On the largest multi-solution group `G22` (`19` tasks, `9+`), `Diversity` improves over `SFT` by `+2.46%` (`9+` bucket remains at `45.99%` normalized recovery).
    2. **Single-solution trade-off**: On the largest `|H_O|=1` group `G02` (`13` tasks), `Diversity-GRPO` scores `76.92%` vs `SFT` `92.31%` (offset by `G03` and `G04`, tying `|H_O|=1` at `85.29%`).
    3. **Cross-run variation reference**: Evaluating the earlier v1.0 run on this exact same 5-node OOD dataset yielded `SFT = 60.90%` (Pooled `33.36%`) and `GRPO = 61.64%` (Pooled `29.13%`), compared to v2.1's `SFT = 56.97%` (Pooled `28.28%`) and `Diversity = 61.70%` (Pooled `31.38%`). Although v1.0 and v2.1 differed in 4-node training splits and SFT schedules, cross-run variation (~`±4pp`) is comparable in magnitude to Stage 2's `+4.73%` Task Delta.

<p align="center">
  <img src="results_analysis/figures/fig3_ood_transfer.png" width="54%" alt="OOD 5-Node Transfer"/>
  <img src="results_analysis/figures/fig4_training_curves.png" width="44%" alt="Training and Ablation Curves"/>
</p>

---

### 2.4 Cross-Domain Extension: Boolean Genetic Interaction & 3D Voxel Reconstruction

#### 2.4.1 Boolean Genetic Interaction ([id_boolean_stats.md](results_analysis/id_boolean_stats.md), [ood_boolean_stats.md](results_analysis/ood_boolean_stats.md))

- **Zero-overlap splits (`x/y` symmetry)**: `train` (`66` tasks / `30` classes), `val` (`14` tasks / `8` classes), **ID Test** (`19` tasks / `13` classes, 3 eval seeds, ceiling `69.03%`), and **Full-NOR OOD** ([boolean_full_d2.json](HypoSpace-release/causal/benchmark_obs_count/paper_scale/boolean_full_d2.json), `80` tasks / `53` classes with unseen `NOR`, single eval seed, ceiling `66.76%`).
- **Metric & checkpoint note**: In the official Boolean evaluator, `Uniqueness` counts **all unique syntactically in-space formulas including invalid ones** (so `Uniqueness` exceeds `Recovery`, e.g., `SFT` `82.44%` vs `22.90%`). We therefore also report exact `Valid Dedup = unique_valid / valid`. Furthermore, the default OOD pipeline pairs `Diversity best` (`step 80`) with `Validity final` (`step 96`), so we explicitly report symmetric `best vs best` and `final vs final` ID comparisons below.

| Split | Model | Valid | Uniq *(incl. invalid)* | Valid Dedup (`n` tasks w/ valid) | **Task Macro Rec** | Task Norm Rec | **Group Rec (mean ± std)** |
|---|---|---:|---:|---:|---:|---:|---:|
| ID (`19` tasks, `3` seeds) | `Qwen2.5-7B Base` | 20.70% | 24.22% | 62.18% (`9`) | 7.64% | 11.38% | 7.97% ± 10.03% |
| ID (`19` tasks, `3` seeds) | `SFT-LoRA` | 30.89% | **82.44%** | **79.62%** (`14`) | 22.90% | 34.52% | 25.31% ± 19.04% |
| ID (`19` tasks, `3` seeds) | `Validity-Only (best / step 20)` | 38.94% | 79.42% | 74.80% (`15`) | 24.36% | 36.70% | 27.32% ± 18.58% |
| ID (`19` tasks, `3` seeds) | `Validity-Only (final)` | **59.12%** | 62.77% | 56.40% (`17`) | 27.89% | 41.10% | 29.83% ± 14.21% |
| ID (`19` tasks, `3` seeds) | **`Diversity-GRPO (best / step 80)`** | 54.16% | 73.50% | 68.51% (`18`) | **31.16%** | **46.23%** | **32.98% ± 14.99%** |
| ID (`19` tasks, `3` seeds) | `Diversity-GRPO (final / step 96)` | 48.78% | 73.00% | 65.30% (`15`) | 26.37% | 39.75% | 30.11% ± 19.91% |
| OOD (`80` tasks, `1` seed) | `Qwen2.5-7B Base` | 19.01% | 9.56% | 14.31% (`18`) | 1.80% | 2.76% | 2.09% ± 4.46% |
| OOD (`80` tasks, `1` seed) | `SFT-LoRA` | 28.86% | **58.01%** | **67.36%** (`59`) | 17.11% | 26.22% | 16.72% ± 11.75% |
| OOD (`80` tasks, `1` seed) | `Validity-Only (final)` | **50.21%** | 32.29% | 42.55% (`68`) | 16.19% | 24.66% | 16.21% ± 9.57% |
| OOD (`80` tasks, `1` seed) | `Diversity-GRPO (best)` | 43.36% | 37.79% | 52.10% (`70`) | **17.66%** | **26.86%** | **17.86% ± 9.19%** |

- **Full balanced Boolean takeaways**:
  - **vs `SFT-LoRA`**: `Diversity-GRPO (best)` significantly improves **Validity** on both ID (**`+23.27%`**, $p = 0.0005$) and OOD (**`+14.50%`**, $p < 10^{-4}$). On **Recovery**, ID shows `+8.26%` Task / `+7.66%` Group (CI `[+1.65%, +13.74%]`, but exact sign test **$p = 0.5488$**), whereas **OOD Recovery differs negligibly (`+0.55%` Task / `+1.13%` Group, $p = 0.7359$, CI `[-1.57%, +3.97%]`, non-significant)**. Moreover, `Diversity-GRPO` **significantly lowers overall `Uniqueness` vs SFT** on both ID (**`-8.94%`**, CI `[-17.00%, -1.71%]`) and OOD (**`-20.22%`**, $p < 10^{-4}$, CI `[-25.55%, -12.55%]`).
  - **vs `Validity-Only` (paired & symmetric checks)**: `Validity-Only` **also increased ID Recovery** (`22.90% → 27.89%`), refuting the hypothesis that validity-only RL always collapses Recovery. Neither asymmetric (`diversity best - validity_final`: ID CI `[-1.59%, +7.96%]`, OOD CI `[-0.75%, +4.06%]`) nor symmetric Recovery comparisons (`best vs best`: Task `+6.81%`, Group `+5.66%`, $p = 0.0654$, CI `[-0.03%, +10.99%]`; `final vs final`: Task `-1.52%`, Group `+0.27%`, $p = 0.7539$, CI `[-6.23%, +6.36%]`, with `diversity_final` `26.37%` < `validity_final` `27.89%`) reach statistical significance. Only OOD `Uniqueness` significantly favors `Diversity best` over `Validity final` (`+5.50%` Task / `+6.32%` Group, $p = 0.0022$, CI `[+2.14%, +10.75%]`).

<p align="center">
  <img src="results_analysis/figures_boolean/fig1_id_ablation.png" width="49%" alt="Boolean ID Ablation"/>
  <img src="results_analysis/figures_boolean/fig4_training_curves.png" width="49%" alt="Boolean Training Curves"/>
</p>

#### 2.4.2 3D Voxel Reconstruction ([id_3d_stats.md](results_analysis/id_3d_stats.md), [ood_3d_b4_stats.md](results_analysis/ood_3d_b4_stats.md))

- **Zero-overlap splits (`D4` symmetry)**: `27` 1–3 block classes into `train` (`16` classes / `73` tasks), `val` (`4` classes / `16` tasks), **ID Test** (`38` tasks / `7` D4 classes, 3 eval seeds, ceiling `64.81%`), and **4-Block OOD** ([3d_g3_h3_b4_ood.json](HypoSpace-release/3d/datasets/3d_g3_h3_b4_ood.json), `23` tasks / `23` classes, single eval seed, ceiling `63.44%`).

| Split | Model | Valid | **Task Macro Rec** | Task Norm Rec | **Group Rec (mean ± std)** | Group Norm Rec |
|---|---|---:|---:|---:|---:|---:|
| ID (`38` tasks, `3` seeds) | `Qwen2.5-7B Base` | 5.10% | 1.59% | 2.41% | 1.50% ± 1.99% | 2.23% ± 2.84% |
| ID (`38` tasks, `3` seeds) | `SFT-LoRA` (`= Diversity best`, `best_step=0`) | 99.77% | 61.31% | 94.39% | **63.38% ± 7.95%** | **96.91% ± 8.78%** |
| ID (`38` tasks, `3` seeds) | `Validity-Only (best / step 90)` | **100.00%** | 61.08% | 94.03% | 63.12% ± 8.17% | 96.50% ± 9.14% |
| ID (`38` tasks, `3` seeds) | `Validity-Only (final)` | **100.00%** | 60.36% | 92.96% | 62.39% ± 7.43% | 95.42% ± 8.26% |
| ID (`38` tasks, `3` seeds) | `Diversity-GRPO (final / step 95)` | **100.00%** | **61.70%** | **95.23%** | 61.82% ± 1.70% | 94.80% ± 2.62% |
| OOD (`23` tasks, `1` seed) | `Qwen2.5-7B Base` | 35.59% | 3.92% | 6.18% | 3.92% ± 2.16% | 6.18% ± 3.40% |
| OOD (`23` tasks, `1` seed) | `SFT-LoRA` (`= Diversity best`) | **100.00%** | 52.50% | 82.75% | 52.50% ± 2.63% | 82.75% ± 4.14% |
| OOD (`23` tasks, `1` seed) | `Validity-Only (final)` | **100.00%** | **53.14%** | **83.76%** | **53.14% ± 4.51%** | **83.76% ± 7.10%** |

- **Ceiling saturation finding**: On 3D, Stage 1 SFT reaches `99.77%` Valid and **`96.91%` Group Norm Recovery** on ID (`82.75%` Norm Recovery zero-shot on 4-Block OOD), leaving negligible headroom below the exact i.i.d. uniform optimum (`max KL = 0.0068`). Consequently, `Diversity best` reverts to `SFT` (`best_step = 0`, zero Stage 2 gain). The small ID difference against `Validity-Only final` (`+0.94%` Task / `+0.99%` Group) is strictly `SFT vs Validity-final` (descriptive at `n = 7` groups) rather than an RL diversity effect; `diversity_final` scores slightly higher in Task Macro Recovery (`61.70%`) but lower in Group Recovery (`61.82%` vs `63.38%`), and on OOD `Validity-final` (`53.14%`) is slightly above SFT (`52.50%`, n.s.).

<p align="center">
  <img src="results_analysis/figures_3d/fig1_id_ablation.png" width="49%" alt="3D ID Ablation"/>
  <img src="results_analysis/figures_3d/fig3_ood_transfer.png" width="49%" alt="3D OOD Transfer"/>
</p>

#### 2.4.3 Cross-Domain Synthesis

Stage 1 balanced SFT is the primary driver of Recovery across all three HypoSpace domains. Stage 2 RL's main consistent measurable effect is improving **Validity**, whereas its additional contribution to **Recovery** depends on how far Stage 1 SFT sits below the exact i.i.d. uniform ceiling: zero on 3D (where SFT already reaches ~97% of the ceiling), and small, directionally positive, yet statistically fragile on Causal and Boolean. Moreover, Boolean shows that validity-only RL can also raise ID Recovery, and that adding the diversity penalty reduces overall Uniqueness relative to SFT.

---

## 3. Method Architecture, Positioning & Discussion

1. **Stage 1 — Epoch-Rotating Balanced SFT-LoRA** ([train_sft.py](HypoSpace-release/causal/grpo_lora/train_sft.py)): Zero-shot 7B fails largely due to low syntactic/constraint validity (`29.47%` on Causal ID, `20.70%` on Boolean ID, `5.10%` on 3D ID). Stage 1 uses prompt loss masking, edge/operand shuffling, and deterministic epoch-rotating target slicing ($\min(K, 8)$ per task per epoch) so all ground truths are covered evenly without overweighting large-$|H_O|$ tasks. **Theoretical note**: equal-weight MLE over exhaustive ground truths $H_O$ already implicitly fits the **uniform distribution over $H_O$** — which, by concavity of $\sum_h [1 - (1 - p_h)^K]$, is the exact expectation-optimal Fixed-Prompt sampling distribution.
2. **Stage 2 — Verifier-Guided Diversity-GRPO-LoRA** ([train_grpo.py](HypoSpace-release/causal/grpo_lora/train_grpo.py), [verifier_reward.py](HypoSpace-release/causal/grpo_lora/verifier_reward.py)): Under unified `SamplingConfig(T=0.9, top_p=0.95)`, we sample $G=8$ completions per task and score them with a strictly monotonic, order-invariant hierarchical inverse-frequency reward:
   $$r_i = \begin{cases}
   -1.0, & \text{unparseable / structurally invalid} \\
   -0.7 + 0.3 \times \text{ObsMatchRatio}(g_i, O), & \text{structurally valid but violating observations } \in [-0.7, -0.4] \\
   0.1 + \frac{0.9 + 0.5 \cdot \bar{d}_{\text{Jaccard}}(g_i, \mathcal{U}\setminus\{g_i\})}{\text{count}(g_i)}, & \text{valid hypothesis satisfying all observations } (\ge +0.1)
   \end{cases}$$
   Because Stage 1 balanced SFT already approximates the uniform distribution where it has sufficient valid probability mass, Stage 2 empirically acts as a **modest post-SFT correction** (`max step KL = 0.020` on Causal, `best_policy` at `step 20` vs `final` `+0.18pp` on val) focused primarily on shifting mass from invalid to valid outputs and partially correcting residual generalization bias.
3. **Related Work & Structural Scale Mismatch (`G < K`)**: Set-level inverse-frequency + similarity rewards within GRPO share core principles with recent quality–diversity RL methods (e.g., *DARLING*-style diversity-aware GRPO). Rather than claiming novelty for the reward functional form itself, our contribution lies in coupling exact domain verifiers, a strict validity-over-diversity hierarchy, and zero-isomorphism-leakage group evaluation across HypoSpace. On large hypothesis spaces (`9+` bucket, $|H_O| \gg 8$), our training rollout size $G = 8$ is substantially smaller than the evaluation budget $K = |H_O|$, providing only coarse set-level diversity feedback and helping explain the limited gains on `9+` tasks — scaling $G$ or introducing group-conditioned exploration remains an open direction.

---

## 4. Limitations

- **Access to exhaustive solver-enumerated ground truths**: Both Stage 1 SFT and Stage 2 filtering depend on datasets containing solver-enumerated exhaustive valid hypotheses during training; in open-ended real-world scientific discovery, exhaustive GTs are unavailable.
- **Small group sample sizes & low statistical power**: Zero-isomorphism-leakage splits leave `5` groups on Causal ID (`n = 76`), `7` on 3D ID (`n = 38`), and `13` on Boolean ID (`n = 19`). With $\le 7$ clusters, cluster bootstrap underestimates variance and exact sign tests have minimal power.
- **Single training run, cross-run variance, and single-seed OOD evaluation**: Each model was trained once; comparing Causal v1.0 (`SFT 60.90%` / `GRPO 61.64%`) vs v2.1 (`SFT 56.97%` / `Diversity 61.70%`) on the identical 5-node OOD set suggests ~`±4pp` run-to-run variation comparable to v2.1's `+4.73%` Task Delta. Furthermore, Causal OOD group significance is sensitive to single-task groups (`+12.29%` overall vs `+6.39%`, CI `[-1.67%, +14.79%]` when `n_tasks >= 2`).
- **Boolean checkpoint selection asymmetry & `Uniqueness` loss**: Default server evaluation paired `Diversity best` with `Validity final` on OOD; under symmetric ID evaluation (`best vs best`, `final vs final`), Diversity vs Validity Recovery differences are not statistically significant, and `Diversity-GRPO` lowers overall `Uniqueness` relative to SFT.
- **DeepSeek temperature confound & 3D/Boolean closed-API scope**: `DeepSeek-Flash` used `T = 0.7` vs Qwen `T = 0.9` (temperature directly affects diversity and Recovery), and closed-API Fixed-Prompt baselines on the zero-leakage 3D and Boolean splits are left as future work (Section 2.4 compares our open-weight ablations directly).
- **No baseline on `causal_n5_gt160`**: Only `Diversity-GRPO-LoRA` was run on the 7-task `|H_O|=160` subset (stress-test reference only).

---

## 5. Repository Structure & Quick Start

```text
.
├── README.md / README_ZH.md                   # Bilingual project overview (all 3 domains)
├── results_analysis/
│   ├── group_stats.py / generate_charts.py    # Causal v2.1 stats & charts -> figures/
│   ├── stats_3d.py / generate_charts_3d.py    # 3D D4-group stats & charts -> figures_3d/
│   ├── stats_boolean.py / generate_charts_boolean.py  # Boolean stats & charts -> figures_boolean/
│   ├── id_v2_stats.md / ood_n5_v2_stats.md    # Causal v2.1 statistical reports (.md / .json)
│   ├── id_3d_stats.md / ood_3d_b4_stats.md    # 3D statistical reports (.md / .json)
│   ├── id_boolean_stats.md / ood_boolean_stats.md  # Boolean statistical reports (.md / .json)
│   └── figures{,_3d,_boolean}/               # 4 publication figures per domain
├── qwen7b_v2_results_and_logs/                # Causal v2.1 RTX 5090 evaluation JSONs & logs
├── qwen7b_results_and_logs/                   # Archived Causal v1.0 baseline logs
├── qwen7b_3d_results_and_logs/                # 3D RTX 5090 evaluation JSONs & training logs
├── qwen7b_boolean_results_and_logs/           # Boolean RTX 5090 evaluation JSONs & training logs
└── HypoSpace-release/
    ├── causal/grpo_lora/                      # Causal SFT + GRPO-LoRA & 10-stage server pipeline
    ├── 3d/grpo_lora/                          # 3D Voxel Reconstruction pipeline & 10-stage server script
    └── boolean/grpo_lora/                     # Boolean Genetic Interaction pipeline & server script
```

### Quick Reproduction Commands

```bash
# 1. Unit tests & zero-isomorphism-leakage split checks across all 3 domains (CPU)
python HypoSpace-release/causal/grpo_lora/tests/test_verifier_reward.py
python HypoSpace-release/causal/grpo_lora/data_utils.py
python -m unittest discover -s HypoSpace-release/3d/grpo_lora/tests -v
python HypoSpace-release/3d/grpo_lora/data_utils.py
python -m unittest discover -s HypoSpace-release/boolean/grpo_lora/tests -v
python HypoSpace-release/boolean/grpo_lora/data_utils.py

# 2. Recompute Causal v2.1 ID & OOD cluster-bootstrap reports & regenerate all 4 figures
python results_analysis/group_stats.py --model diversity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_diversity_seed*.json --model validity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_validity_seed*.json --model sft=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_sft_seed*.json --model base=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_base_seed*.json --deepseek_baseline HypoSpace-release/causal/benchmark_obs_count/full_repo_causal_n4_obs3_deepseek.json --splits_path HypoSpace-release/causal/grpo_lora/splits/causal_splits.json --pairs diversity:sft --pairs diversity:validity --pairs diversity:deepseek --pairs sft:deepseek --pairs diversity:base --pairs sft:base --output_md results_analysis/id_v2_stats.md --output_json results_analysis/id_v2_stats.json

python results_analysis/group_stats.py --model diversity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_grpo_diversity_seed33550336.json --model validity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_grpo_validity_seed33550336.json --model sft=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_sft_seed33550336.json --model base=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_base_seed33550336.json --dataset_path HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json --pairs diversity:sft --pairs diversity:validity --pairs diversity:base --pairs sft:base --output_md results_analysis/ood_n5_v2_stats.md --output_json results_analysis/ood_n5_v2_stats.json

python results_analysis/generate_charts.py

# 3. Full GPU training & multi-seed evaluation pipelines (serial execution on one RTX 5090, from HypoSpace-release/)
bash causal/grpo_lora/run_server_all.sh
bash 3d/grpo_lora/run_server_all.sh
bash boolean/grpo_lora/run_server_all.sh
```
For full RTX 5090 deployment details, see [causal/grpo_lora/RUN_GUIDE.md](HypoSpace-release/causal/grpo_lora/RUN_GUIDE.md), [3d/grpo_lora/RUN_GUIDE.md](HypoSpace-release/3d/grpo_lora/RUN_GUIDE.md), and [boolean/grpo_lora/RUN_GUIDE.md](HypoSpace-release/boolean/grpo_lora/RUN_GUIDE.md).
