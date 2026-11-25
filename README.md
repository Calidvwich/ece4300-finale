# EEC 4300 Final Project — Breaking Sublinear Coverage in HypoSpace via Two-Stage Diversity-Aware GRPO-LoRA

This repository contains our **EEC 4300** group project on **HypoSpace** (*ICML 2026*, [PMLR Vol. 306](https://proceedings.mlr.press/v306/chen26bz.html)), which evaluates LLMs as samplers over **underdetermined scientific hypothesis spaces** using three metrics: **Validity**, **Uniqueness**, and **Recovery**.

---

## 1. Problem & Motivation

Under HypoSpace's formal **Fixed-Prompt** independent sampling protocol (`--independent-samples`, $N = |H_O|$ queries per task without appending prior proposals into the prompt), even strong closed-source LLMs suffer severe **mode collapse**:
- On the 4-node causal inference benchmark ([n4_obs3_all.json](HypoSpace-release/causal/datasets/n4_obs3_all.json)), **`DeepSeek-Flash`** achieves **99.74%–100% Validity**, but its **Recovery collapses to ~45%–46%** (and drops to **14.34%** when $|H_O| \ge 9$).
- Standard supervised fine-tuning (**SFT-LoRA**) learns formatting and reachability, yet lacks a set-level anti-repetition objective and leaves Validity at **88.29%**.

We propose **Two-Stage Diversity-Aware GRPO-LoRA** on an open-weights 7B model (`Qwen2.5-7B-Instruct`) to directly reshape the sampler's internal distribution—without prompt-history tricks or verifier-feature leakage.

---

## 2. Headline Results (Fixed-Prompt Protocol, $N = |H_O|$)

### 2.1 In-Distribution Causal Test (79 Held-out 4-Node Tasks, 433 Queries)

All models are evaluated on the exact same held-out 79 `observation_set_id`s from [causal_splits.json](HypoSpace-release/causal/grpo_lora/splits/causal_splits.json), paired directly against our recorded `DeepSeek-Flash` baseline ([full_repo_causal_n4_obs3_deepseek.json](HypoSpace-release/causal/benchmark_obs_count/full_repo_causal_n4_obs3_deepseek.json)):

| Model / Method | Parse | Macro Valid | Macro Uniq | **Macro Recovery** | **Pooled Recovery** | vs. DeepSeek-Flash |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B Base` (Zero-shot) | 99.84% | 18.35% | 8.13% | 8.13% | 3.93% | -38.19% |
| `DeepSeek-Flash` (Closed API) | **100.00%** | **100.00%** | 46.32% | 46.32% | 25.87% | Baseline |
| `Qwen2.5-7B + SFT-LoRA` (Stage 1) | 99.37% | 88.29% | 58.93% | 58.93% | **50.58%** | +12.61% |
| **`Qwen2.5-7B + Diversity-GRPO-LoRA` (Ours)** | **100.00%** | **94.03%** | **61.62%** | **61.62%** | 49.88% | **+15.30%** |

- **Paired significance tests (79 tasks)**:
  - `Diversity-GRPO-LoRA` vs. `DeepSeek-Flash`: $\Delta = +15.29\%$ (**39 wins / 34 ties / 6 losses**), paired $t = 5.116$ ($p = 2.18 \times 10^{-6}$), Wilcoxon $p = 2.25 \times 10^{-6}$.
  - `Diversity-GRPO-LoRA` vs. `Base 7B`: $\Delta = +53.49\%$ (**71 wins / 8 ties / 0 losses**), $p = 6.13 \times 10^{-25}$.

#### Recovery Breakdown by Hypothesis Space Size $|H_O|$

| Bucket (`\|H_O\|`) | Tasks | `Base 7B` | `DeepSeek-Flash` | `SFT-LoRA` | **`Diversity-GRPO-LoRA`** | GRPO / DeepSeek |
|---|---:|---:|---:|---:|---:|---:|
| `1` | 15 | 26.67% | **100.00%** | 80.00% | **93.33%** | 0.93x |
| `2–4` | 29 | 4.02% | 51.15% | 62.07% | **66.09%** *(Valid 100%)* | **1.29x** |
| `5–8` | 18 | 5.28% | 24.03% | **46.67%** | **44.31%** | **1.84x** |
| **`9+` (Hardest)** | 17 | 1.80% | 14.34% | **47.96%** | **44.32%** | **3.09x** |

<p align="center">
  <img src="results_analysis/figures/fig1_id_ablation.png" width="49%" alt="ID 4-way Ablation"/>
  <img src="results_analysis/figures/fig2_difficulty_scaling.png" width="49%" alt="Difficulty Scaling"/>
</p>

### 2.2 Out-of-Distribution 5-Node Transfer (Trained Only on 3–4 Nodes)

Evaluated zero-shot on unseen 5-node causal systems ([n5_obs5_hybrid80.json](HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json), 80 tasks, 1,418 queries; plus [causal_n5_gt160.json](HypoSpace-release/causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json), 7 tasks with $|H_O|=160$):

| Dataset | Model | Macro Valid | Macro Uniq | **Macro Recovery** | Pooled Recovery | `Bucket 9+` Recovery |
|---|---|---:|---:|---:|---:|---:|
| `n5_obs5_hybrid80` (80 tasks) | `Qwen2.5-7B Base` | 32.50% | 31.56% | 31.56% | 1.90% | 0.00% |
| `n5_obs5_hybrid80` (80 tasks) | `Qwen2.5-7B + SFT-LoRA` | **72.89%** | 60.90% | 60.90% | **33.36%** | **31.39%** |
| `n5_obs5_hybrid80` (80 tasks) | **`Qwen2.5-7B + Diversity-GRPO-LoRA`** | 69.61% | **61.64%** | **61.64%** | 29.13% | 26.90% |
| `causal_n5_gt160` (7 tasks, `\|H_O\|=160`) | **`Qwen2.5-7B + Diversity-GRPO-LoRA`** | 32.41% | 11.96% | 11.96% | 11.96% | 11.96% |

- On `n5_obs5_hybrid80`, `Base 7B` gets **0.00%** on both `5–8` and `9+` buckets, while `Diversity-GRPO-LoRA` improves Macro Recovery by **+30.08%** (53 wins / 22 ties / 5 losses, Wilcoxon $p = 1.44 \times 10^{-6}$).
- **Observation**: Stage 2 GRPO yields clear gains in-distribution (`Valid`: 88.29% → 94.03%, `Macro Recovery`: 58.93% → 61.62%), whereas Stage 1 SFT retains slightly higher Pooled Recovery on large 5-node spaces (33.36% vs. 29.13%), suggesting mild node-scale specialization during 3–4 node RL.

<p align="center">
  <img src="results_analysis/figures/fig3_ood_transfer.png" width="56%" alt="OOD 5-Node Transfer"/>
  <img src="results_analysis/figures/fig4_training_curves.png" width="42%" alt="Training Curves"/>
</p>

---

## 3. Method Architecture

### Why Two Stages?
1. **Stage 1 — Debiased SFT-LoRA Warm-up** ([train_sft.py](HypoSpace-release/causal/grpo_lora/train_sft.py)): Zero-shot 7B produces only 18.35% valid DAGs, which would cause zero-variance reward collapse in pure RL. Stage 1 applies prompt loss masking, uniform sampling over up to `8` ground truths per task, and randomized edge ordering (`shuffle_edges=True`), creating both a strong warm-start and a clean supervised ablation baseline.
2. **Stage 2 — Diversity-Aware GRPO-LoRA** ([train_grpo.py](HypoSpace-release/causal/grpo_lora/train_grpo.py), [verifier_reward.py](HypoSpace-release/causal/grpo_lora/verifier_reward.py)): For each underdetermined task ($|H_O| \ge 2$, 227 training tasks), we sample a group of $G=8$ completions and assign **set-level diversity rewards**:
   - Unparseable / cyclic / invalid-node graph: `-1.0`
   - Valid DAG violating observations: `-0.5 + 0.3 * obs_match_ratio`
   - **Unique valid graph** (first occurrence in group): `1.0 + 0.5 * mean_edge_jaccard_distance(g_i, U_valid \ {g_i})`
   - **Duplicate valid graph** (already seen in group): `-0.3` (direct anti-collapse penalty)

   We optimize the clipped GRPO objective with Schulman's unbiased KL penalty ($\beta = 0.04$) against the frozen Stage 1 reference adapter—mounted on the **same base model** (`"policy"` vs. `"ref"`) with micro-batched gradient accumulation (`micro_batch_size=2`) and `use_reentrant=False` gradient checkpointing so 7B runs cleanly on a single **32GB RTX 5090**.

### Base Model & Portability Note
- `deepseek-flash` is a closed-source API (no weight/gradient access), and `DeepSeek-R1-Distill-Qwen-7B` emits long `<think>` chains that slow online group rollouts. We therefore trained on **`Qwen2.5-7B-Instruct`**.
- Saved LoRA adapters ($\Delta W$) are tied to `Qwen2.5-7B-Instruct` and cannot be loaded onto DeepSeek APIs, while the **Two-Stage Diversity-GRPO-LoRA pipeline is completely model-agnostic** across open-weights causal LMs.

---

## 4. Repository Structure

```text
.
├── README.md                                  # Global project overview (this file)
├── .gitignore                                 # Excludes large binaries (>100MB), logs, caches, local notes
├── results_analysis/
│   ├── generate_charts.py                     # Reproduces all 4 figures directly from saved JSON/JSONL logs
│   └── figures/                               # fig1_id_ablation.png .. fig4_training_curves.png
├── qwen7b_results_and_logs/
│   └── causal/grpo_lora/
│       ├── results/                           # All RTX 5090 evaluation JSONs (ID & OOD)
│       └── outputs/                           # sft_metrics.json & grpo_metrics.jsonl training logs
└── HypoSpace-release/                         # Official benchmark code + teammate experiments + GRPO-LoRA
    ├── README.md                              # Original HypoSpace benchmark README
    ├── 3d/ & boolean/                         # Secondary benchmark domains
    └── causal/
        ├── benchmark_obs_count/               # Teammate's DeepSeek API baselines & ranker/edit experiments
        │   └── FULL_API_ANALYSIS.md           # Detailed DeepSeek-Flash Fixed vs. History diagnostic report
        └── grpo_lora/                         # Our Two-Stage Diversity-Aware GRPO-LoRA implementation
            ├── config.py                      # Unified relative paths & hyperparameters
            ├── data_utils.py                  # Stratified leakage-free split & Fixed-Prompt builder
            ├── verifier_reward.py             # Official-aligned DAG verifier & set-level diversity reward
            ├── train_sft.py                   # Stage 1 SFT-LoRA warm-up
            ├── train_grpo.py                  # Stage 2 Diversity-GRPO-LoRA
            ├── evaluate_lora.py               # Fixed-Prompt evaluator with paired DeepSeek comparison
            ├── run_server_all.sh              # 8-stage Linux GPU pipeline script
            ├── RUN_GUIDE.md                   # Detailed setup & reproduction instructions
            ├── splits/causal_splits.json      # Exact Train (293) / Val (39) / ID Test (79) splits
            └── tests/test_verifier_reward.py  # CPU unit tests
```

*(Note: Files > 100 MB such as `n5_obs1to5_80each.json` and the 154 MB LoRA checkpoint are excluded via `.gitignore`; all evaluation JSONs and training logs are preserved in `qwen7b_results_and_logs/`.)*

---

## 5. Quick Start for Teammates

### 5.1 Verify Verifier, Rewards, and Leakage-Free Splits (CPU, ~1 sec)
```bash
python HypoSpace-release/causal/grpo_lora/tests/test_verifier_reward.py
```

### 5.2 Regenerate All Figures from Saved Results
```bash
python results_analysis/generate_charts.py
```

### 5.3 Full Training & Evaluation Pipeline (Linux GPU, RTX 5090 32GB / RTX 4090 24GB)
See [HypoSpace-release/causal/grpo_lora/RUN_GUIDE.md](HypoSpace-release/causal/grpo_lora/RUN_GUIDE.md) for environment setup, or run directly from `HypoSpace-release/`:
```bash
pip install -r causal/grpo_lora/requirements_lora.txt
bash causal/grpo_lora/run_server_all.sh
```
For baseline API runs (`DeepSeek-Flash`), see [HypoSpace-release/README.md](HypoSpace-release/README.md) and [FULL_API_ANALYSIS.md](HypoSpace-release/causal/benchmark_obs_count/FULL_API_ANALYSIS.md).
