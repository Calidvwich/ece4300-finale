# Boolean SFT-LoRA / GRPO-LoRA 运行指南

## 服务器全量训练与评测

在仓库根目录，用已经配置好 CUDA、PyTorch 和 Boolean 依赖的 Python 环境运行。LoRA 依赖范围沿用 `HypoSpace-release/causal/grpo_lora/requirements_lora.txt`。CPU 测试还使用 NumPy、SymPy、SciPy、PyYAML 和 Requests，画图使用 Matplotlib。

~~~bash
export PYTHON=/opt/envs/torch5090/bin/python
"$PYTHON" -c 'import torch, transformers, peft, sympy; print(torch.cuda.get_device_name())'
RUN_TAG=qwen7b_boolean bash HypoSpace-release/boolean/grpo_lora/run_server_all.sh
~~~

脚本依次执行切分与测试、Base ID 评测、SFT 和 ID 评测、Validity-Only GRPO 和 ID 评测、Diversity GRPO 和 ID 评测、OOD 评测。两个 GRPO 模式都从同一份 SFT adapter 开始，分别写入 `best_policy/` 和 `final_policy/`。ID 对两种 checkpoint 都运行，OOD 使用 Validity-Only final 和 Diversity best。

- `MODEL_NAME` 默认 `Qwen/Qwen2.5-7B-Instruct`，也可指定本地权重目录。`RUN_TAG` 默认 `qwen7b_boolean`。
- ID 评测固定使用种子 `33550336 1 2`。OOD 默认只用 `33550336`，设置 `OOD_MULTI_SEED=1` 后使用全部三个种子。OOD 全集引入 NOR、扩大假设空间，其中 30 个观测等价类也出现在训练集。
- `SKIP_VALIDITY=1` 跳过 Validity-Only 训练以及它的 ID、OOD 评测。显存不足时可设置 `MICRO_BATCH_SIZE=1 GEN_BATCH_SIZE=4`，每题 rollout 仍为 8 条。
- 默认 SFT 2 个 epoch，约 42 个优化步。GRPO 过滤 `|H_O| < 2` 后剩 64 道训练题，每 4 题一步，默认 6 个 epoch，每种奖励 96 步。`SFT_EPOCHS` 和 `GRPO_EPOCHS` 可覆盖默认值。
- 每题验证和评测采样 `|H_O|` 次。Qwen rollout、验证、评测均用 temperature 0.9、top-p 0.95、max new tokens 64。

| 阶段 | 默认生成次数 |
|---|---:|
| GRPO rollout，两种奖励 | 6,144 |
| GRPO validation，含两次 Step 0 | 1,080 |
| ID 测试，6 个变体 × 3 个种子 | 2,232 |
| OOD 测试，4 个变体 × 1 个种子 | 3,900 |
| 合计 | 13,356 |

`OOD_MULTI_SEED=1` 时，OOD 生成增至 11,700 次。RTX 5090 单卡默认全流程建议预留约 2–6 GPU 小时。此值按最多 85 万生成 token、批量解码和 LoRA 前后向开销粗估。实际耗时取决于生成长度、批量和服务器配置，此处未实测 7B。

## DeepSeek Fixed-Prompt 基线

从 `HypoSpace-release/` 运行以下命令。通过环境变量 `HYPOSPACE_DEEPSEEK_API_KEY` 提供密钥。官方 runner 使用 `deepseek-flash` 和 temperature 0.7。Fixed-Prompt 模式用 metadata 顺序排列算子，每题每次调用使用同样的提示词，解析失败也计入一次查询。

~~~bash
cd HypoSpace-release

"$PYTHON" boolean/boolean_benchmark.py \
  --dataset boolean/datasets/boolean_extended_d2_test.json \
  --config boolean/grpo_lora/deepseek_fixed_prompt.yaml \
  --n-samples 19 --query-multiplier 1.0 --seed 33550336 --independent-samples

"$PYTHON" boolean/boolean_benchmark.py \
  --dataset causal/benchmark_obs_count/paper_scale/boolean_full_d2.json \
  --config boolean/grpo_lora/deepseek_fixed_prompt.yaml \
  --n-samples 80 --query-multiplier 1.0 --seed 33550336 --independent-samples
~~~

ID 共 124 次，OOD 共 975 次。沿用 Boolean runner 的 YAML 输出和 checkpoint 接口。`benchmark.output_pattern` 以数据集名区分两个 JSON，`benchmark.checkpoint` 下也按数据集名建子目录。

| 数据 | 输出相对于 `HypoSpace-release/` | checkpoint 子目录 |
|---|---|---|
| ID | `boolean/grpo_lora/results/boolean_extended_d2_test_deepseek_fixed_prompt.json` | `boolean/grpo_lora/checkpoints/fixed_prompt/boolean_extended_d2_test/` |
| OOD | `boolean/grpo_lora/results/boolean_full_d2_deepseek_fixed_prompt.json` | `boolean/grpo_lora/checkpoints/fixed_prompt/boolean_full_d2/` |

## 统计与作图

以下命令从仓库根目录运行。模型参数支持多 seed 文件的 glob，脚本先按题平均各个 seed，再按观测等价类汇总。存在 DeepSeek 文件时传入相应参数。跳过 Validity-Only 时省略它的 `--model` 和 `--validity-log`。

~~~bash
B=HypoSpace-release/boolean/grpo_lora
R="$B/results"
O="$B/outputs"
T=qwen7b_boolean

"$PYTHON" results_analysis/stats_boolean.py \
  --model "base=$R/${T}_id_base_seed*.json" \
  --model "sft=$R/${T}_id_sft_seed*.json" \
  --model "validity_best=$R/${T}_id_validity_best_seed*.json" \
  --model "validity_final=$R/${T}_id_validity_final_seed*.json" \
  --model "diversity=$R/${T}_id_diversity_seed*.json" \
  --model "diversity_final=$R/${T}_id_diversity_final_seed*.json" \
  --deepseek_baseline "$R/boolean_extended_d2_test_deepseek_fixed_prompt.json" \
  --output_md "$R/id_stats.md" --output_json "$R/id_stats.json"

"$PYTHON" results_analysis/stats_boolean.py \
  --dataset_path HypoSpace-release/causal/benchmark_obs_count/paper_scale/boolean_full_d2.json \
  --model "base=$R/${T}_ood_base_seed*.json" \
  --model "sft=$R/${T}_ood_sft_seed*.json" \
  --model "validity_final=$R/${T}_ood_validity_final_seed*.json" \
  --model "diversity=$R/${T}_ood_diversity_seed*.json" \
  --deepseek_baseline "$R/boolean_full_d2_deepseek_fixed_prompt.json" \
  --output_md "$R/ood_stats.md" --output_json "$R/ood_stats.json"

"$PYTHON" results_analysis/generate_charts_boolean.py \
  --id-stats "$R/id_stats.json" --ood-stats "$R/ood_stats.json" \
  --sft-metrics "$O/${T}_sft/sft_metrics.json" \
  --diversity-log "$O/${T}_grpo_diversity/grpo_metrics.jsonl" \
  --validity-log "$O/${T}_grpo_validity/grpo_metrics.jsonl"
~~~

报告含 task 与等价类宏平均、难度分桶、均匀独立采样参考值、归一化 Recovery、配对双侧精确符号检验，以及 10,000 次等价类 bootstrap 的 95% 区间。种子为 `33550336`。原始 GT 与官方 verifier 存在的差异记录在 [HANDOFF.md](HANDOFF.md)。
