# EEC 4300 Final Project — Breaking Sublinear Coverage in HypoSpace via Two-Stage Diversity-Aware GRPO-LoRA

> **Language**: **English** | [简体中文](README_ZH.md)

This repository contains our **EEC 4300** group project on **HypoSpace** (*ICML 2026*, [PMLR Vol. 306](https://proceedings.mlr.press/v306/chen26bz.html)), which evaluates LLMs as samplers over **underdetermined scientific hypothesis spaces** using three metrics: **Validity**, **Uniqueness**, and **Recovery**.

---

## 1. Problem & Motivation

Under HypoSpace's formal **Fixed-Prompt** independent sampling protocol (`--independent-samples`, $N = |H_O|$ queries per task without appending prior proposals into the prompt), even strong closed-source LLMs suffer severe **mode collapse**:
- On the 4-node causal inference benchmark ([n4_obs3_all.json](HypoSpace-release/causal/datasets/n4_obs3_all.json)), **`DeepSeek-Flash`** achieves **100.00% Validity**, but its **Recovery drops to 38.34%** on our unseen-structure test set (and collapses to **18.52%** when $|H_O| \ge 9$).
- Standard supervised fine-tuning (**SFT-LoRA**) learns formatting and causal reachability, yet lacks a set-level anti-repetition objective; meanwhile, a naive **Validity-Only GRPO** ablation collapses toward a narrow set of safe solutions (`Val Valid`: 77.52% → 96.27%, while `Val Group Recovery` drops 55.82% → 52.17%).

We propose **Two-Stage Diversity-Aware GRPO-LoRA** on an open-weights 7B model (`Qwen2.5-7B-Instruct`) under a **zero-isomorphism-leakage split** and unified sampling configuration (`T = 0.9, top_p = 0.95`), reshaping the sampler's internal distribution without prompt-history tricks or verifier-feature leakage.

---

## 2. Headline Results (v2.1 Zero-Isomorphism-Leakage Protocol)

### 2.1 Zero-Isomorphism-Leakage Split & Theoretical Ceilings

All 392 tasks in `n4_obs3_all.json` reduce to **22 node-permutation ($S_4$) isomorphism equivalence classes** under canonical observation signatures (`canonical_isomorphism_key` in [data_utils.py](HypoSpace-release/causal/grpo_lora/data_utils.py)). We split **entire equivalence classes** into:
- **Train** (`13` groups, `256` `n=4` + `19` `n=3` = `275` tasks)
- **Val** (`4` groups, `60` tasks; `48` stratified tasks = `12/group` used for balanced validation)
- **Unseen-Structure ID Test** (`5` groups, `76` tasks, `400` queries)

Because Fixed-Prompt draws $K = |H_O|$ independent samples, the finite-sample **theoretical recovery ceiling** is $1 - (1 - 1/K)^K$ (`70.38%` macro across the 76 ID tasks; `81.41%` across the 80 OOD 5-node tasks). We report both **raw** and **ceiling-normalized** recovery (`recovery_norm`), together with **isomorphism-group cluster bootstrap 95% CIs (10,000 resamples)** via [group_stats.py](results_analysis/group_stats.py).

### 2.2 Unseen-Structure 4-Node ID Test (76 Tasks / 5 Groups, 3-Seed Mean)

| Model / Method | Seeds | Parse | Macro Valid | **Macro Rec (= Uniq)** | Task Norm Rec | **Group Rec (mean ± std)** | Group Norm Rec | **Pooled Rec** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B Base` (Zero-shot) | 3 | 100.00% | 29.47% | 11.29% | 14.97% | 18.34% ± 22.45% | 21.26% ± 20.89% | 8.42% |
| `DeepSeek-Flash` (Closed API) | 1 | **100.00%** | **100.00%** | 38.34% | 52.47% | 47.80% ± 33.61% | 59.40% ± 31.47% | 27.00% |
| `Qwen2.5-7B + SFT-LoRA` (Stage 1) | 3 | 97.69% | 86.71% | 55.63% | 78.70% | 61.81% ± 17.15% | 81.26% ± 6.98% | 52.33% |
| `Validity-Only GRPO` (Stage 2a Ablation)* | 3 | 97.69% | 86.71% | 55.63% | 78.70% | 61.81% ± 17.15% | 81.26% ± 6.98% | 52.33% |
| **`Diversity-GRPO-LoRA` (Stage 2b Ours)** | 3 | 98.42% | **91.91%** | **58.38%** | **82.50%** | **64.37% ± 16.88%** | **84.78% ± 8.10%** | **53.50%** |

*\*Note on Stage 2a (`Validity-Only GRPO`): Without diversity rewards, validation validity rose from `77.52%` to `96.27%`, while group-mean validation recovery monotonically degraded (`55.82%` → `52.17%`), triggering `best_step == 0` protection so `best_policy/` retained the SFT initialization.*

- **Statistical comparison ([id_v2_stats.md](results_analysis/id_v2_stats.md))**:
  - `Diversity-GRPO-LoRA` vs. `DeepSeek-Flash`: **+20.04%** Task Recovery (`57` wins / `11` ties / `8` losses), **+16.57%** Group Recovery (Cluster Bootstrap 95% CI **`[+2.71%, +29.52%]`**, Group Norm **+25.38%**, CI `[+6.27%, +44.49%]`).
  - `Diversity-GRPO-LoRA` vs. `SFT-LoRA`: Outperforms SFT on **3/3 evaluation seeds** (`57.75% vs 54.20%`, `58.13% vs 54.33%`, `59.27% vs 58.35%`), improving Task Validity by **+5.20%** (Group **+4.46%**, 95% CI **`[+0.42%, +8.73%]`**) and Task Recovery by **+2.75%** (Group **+2.56%**, 95% CI `[-0.31%, +5.43%]`; note that with $n = 5$ test isomorphism groups on 4-node tasks, group-level statistical power is inherently limited).

#### Recovery Breakdown by Hypothesis Space Size $|H_O|$

| Bucket (`\|H_O\|`) | Tasks (Groups) | Theoretical Ceiling | `Base 7B` | `DeepSeek-Flash` | `SFT-LoRA` | **`Diversity-GRPO-LoRA`** | Norm Rec (Ours) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 4 (1) | 100.00% | 58.33% | **100.00%** | 91.67% | **91.67%** | 91.67% |
| `2–4` | 36 (2) | 71.91% | 10.03% | 51.85% | 57.10% | **63.43%** | **88.22%** |
| `5–8` | 24 (1) | 65.64% | 8.51% | 17.71% | **47.92%** | 46.53% | 70.88% |
| **`9+` (Hardest)** | 12 (1) | 65.36% | 4.94% | 18.52% | 54.63% | **55.86% (`3.0x` DS)** | **85.48%** |

<p align="center">
  <img src="results_analysis/figures/fig1_id_ablation.png" width="49%" alt="Unseen-Structure ID Ablation"/>
  <img src="results_analysis/figures/fig2_difficulty_scaling.png" width="49%" alt="Difficulty Scaling vs Ceiling"/>
</p>

### 2.3 Out-of-Distribution 5-Node Transfer (80 Tasks / 22 Structure Groups)

Evaluated zero-shot on unseen 5-node causal systems ([n5_obs5_hybrid80.json](HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json), 80 tasks, 22 structure groups, Ceiling `81.41%`; plus [causal_n5_gt160.json](HypoSpace-release/causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json), 7 tasks with $|H_O|=160$):

| Dataset | Model | Macro Valid | **Macro Rec (= Uniq)** | **Group Rec (mean ± std)** | Group Norm Rec | **Pooled Rec** | `Bucket 9+` Rec |
|---|---|---:|---:|---:|---:|---:|---:|
| `n5_obs5_hybrid80` (80 tasks) | `Qwen2.5-7B Base` | 28.14% | 26.58% | 17.78% ± 33.72% | 18.71% ± 33.90% | 1.69% | 0.07% |
| `n5_obs5_hybrid80` (80 tasks) | `SFT-LoRA` (`= Validity-GRPO*`) | 67.54% | 56.97% | 50.92% ± 28.47% | 61.69% ± 26.17% | 28.28% | 26.43% |
| `n5_obs5_hybrid80` (80 tasks) | **`Diversity-GRPO-LoRA` (Ours)** | **70.33%** | **61.70%** | **63.21% ± 26.63%** | **78.17% ± 24.67%** | **31.38%** | **29.28%** |
| `causal_n5_gt160` (7 tasks, `\|H_O\|=160`) | **`Diversity-GRPO-LoRA` (Ours)** | 19.29% | 10.71% | — | — | 10.71% | 10.71% |

- **Strong OOD statistical significance across 22 structure groups ([ood_n5_v2_stats.md](results_analysis/ood_n5_v2_stats.md))**:
  - `Diversity-GRPO-LoRA` vs. `SFT-LoRA`: **+4.73%** Task Macro Recovery, **+3.10%** Pooled Recovery, and **+12.29% ± 24.88%** Structure-Group Recovery (**12 wins / 8 ties / 2 losses**, exact two-sided binomial sign test **$p = 0.0129$**, Cluster Bootstrap 95% CI **`[+3.50%, +23.68%]`**). Normalized Group Recovery gains **+16.49%** (95% CI **`[+5.01%, +31.49%]`**), winning across every multi-solution bucket (`2–4`: `63.24% vs 47.06%`, `5–8`: `47.92% vs 41.67%`, `9+`: `29.28% vs 26.43%`).
  - `Diversity-GRPO-LoRA` vs. `Base 7B`: **+45.43%** Group Recovery (19 wins / 3 ties / 0 losses, sign test **$p < 10^{-4}$**, 95% CI `[+33.81%, +57.77%]`).

<p align="center">
  <img src="results_analysis/figures/fig3_ood_transfer.png" width="54%" alt="OOD 5-Node Transfer"/>
  <img src="results_analysis/figures/fig4_training_curves.png" width="44%" alt="Training and Ablation Curves"/>
</p>

---

## 3. Method Architecture (v2.1)

1. **Stage 1 — Epoch-Rotating Debiased SFT-LoRA Warm-up** ([train_sft.py](HypoSpace-release/causal/grpo_lora/train_sft.py)): Zero-shot 7B achieves only 29.47% valid DAGs on unseen 4-node structures. Stage 1 uses prompt loss masking, edge-order shuffling, and deterministic epoch-rotating target slicing ($\min(K, 8)$ per task per epoch), reaching **100.00% coverage** over all `1,353` ground truths (`275` training tasks) in 2 epochs without overweighting large-$|H_O|$ tasks.
2. **Stage 2 — Permutation-Invariant Diversity-Aware GRPO-LoRA** ([train_grpo.py](HypoSpace-release/causal/grpo_lora/train_grpo.py), [verifier_reward.py](HypoSpace-release/causal/grpo_lora/verifier_reward.py)): Under unified `SamplingConfig(T=0.9, top_p=0.95, max_new_tokens=64)`, we sample $G=8$ completions per underdetermined task and score them with a strictly monotonic, order-invariant inverse-frequency diversity reward:
   $$r_i = \begin{cases}
   -1.0, & \text{unparseable / cyclic / invalid node or edge budget} \\
   -0.7 + 0.3 \times \text{ObsMatchRatio}(g_i, O), & \text{valid DAG violating observations } \in [-0.7, -0.4] \\
   0.1 + \frac{0.9 + 0.5 \cdot \bar{d}_{\text{Jaccard}}(g_i, \mathcal{U}\setminus\{g_i\})}{\text{count}(g_i)}, & \text{valid graph satisfying all observations } (\ge +0.1)
   \end{cases}$$
   Training disables dropout (`dropout=0` while preserving `use_reentrant=False` checkpointing), accumulates gradients over 4 tasks/step, uses Cosine LR (`1e-5`), Schulman unbiased KL ($\beta = 0.06$) between dual adapters (`"policy"` vs. frozen `"ref"`) on one 7B backbone, and selects `best_policy` strictly (`>`) over Step-0 SFT initialization using balanced group-mean validation recovery (`12` tasks/group).

---

## 4. Repository Structure & Quick Start

```text
.
├── README.md / README_ZH.md                   # Bilingual project overview
├── results_analysis/
│   ├── group_stats.py                         # Isomorphism-group stats, ceilings & cluster bootstrap CI
│   ├── generate_charts.py                     # Reproduces all 4 figures from v2.1 JSONs
│   ├── id_v2_stats.md / .json                 # Unseen-structure 4-node statistical report
│   ├── ood_n5_v2_stats.md / .json             # 5-node OOD statistical report
│   └── figures/                               # fig1_id_ablation.png .. fig4_training_curves.png
├── qwen7b_v2_results_and_logs/                # Final v2.1 RTX 5090 evaluation JSONs & training logs
├── qwen7b_results_and_logs/                   # Archived v1.0 baseline logs
└── HypoSpace-release/causal/grpo_lora/        # Full implementation & 10-stage server pipeline
```

### Quick Reproduction Commands

```bash
# 1. Unit tests & zero-isomorphism-leakage split verification (CPU, ~1 sec)
python HypoSpace-release/causal/grpo_lora/tests/test_verifier_reward.py
python HypoSpace-release/causal/grpo_lora/data_utils.py

# 2. Recompute cluster-bootstrap statistical reports & regenerate all 4 figures
python results_analysis/group_stats.py --model diversity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_diversity_seed*.json --model validity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_validity_seed*.json --model sft=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_sft_seed*.json --model base=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_base_seed*.json --deepseek_baseline HypoSpace-release/causal/benchmark_obs_count/full_repo_causal_n4_obs3_deepseek.json --splits_path HypoSpace-release/causal/grpo_lora/splits/causal_splits.json --output_md results_analysis/id_v2_stats.md --output_json results_analysis/id_v2_stats.json
python results_analysis/generate_charts.py

# 3. Full 10-stage GPU training & multi-seed evaluation pipeline (from HypoSpace-release/)
bash causal/grpo_lora/run_server_all.sh
```
For full RTX 5090 deployment details, see [RUN_GUIDE.md](HypoSpace-release/causal/grpo_lora/RUN_GUIDE.md).
