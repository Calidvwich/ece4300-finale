# Boolean GRPO-LoRA 交付说明

实现基于 GitHub 默认分支 `feat/diversity-grpo-lora` 的 `75cf9f5`。训练和评测入口放在 `boolean/grpo_lora/`，测试集按官方格式导出到 `boolean/datasets/`。配套命令见 [RUN_GUIDE.md](RUN_GUIDE.md)。

## 切分与查询预算

Extended 有 78 题和 51 个变量交换等价类，使用种子 `33550336` 按兼容机制数分桶切分。Basic 35 道题中有 14 道的观测等价类落入 Extended val/test，这 14 道被移除，余下 21 道并入 train。每个 split 内通过 `数据集名:observation_set_id` 区分跨数据集重名。

| 切分 | 题数 | 观测等价类 | 按 N = K 查询 | K=1 | K=2–4 | K=5–8 | K≥9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train（含 Basic） | 66 | 30 | 394 | 2 | 22 | 28 | 14 |
| Val | 14 | 8 | 90 | 0 | 5 | 5 | 4 |
| ID Test | 19 | 13 | 124 | 0 | 9 | 6 | 4 |
| OOD Full | 80 | 53 | 975 | 2 | 12 | 27 | 39 |

Train、val、test 两两等价类交集为零。Extended 的 K=1 只有一个等价类，分到 train。每个 K 的完整频数见 `splits/boolean_splits.json` 的 `metadata.counts`。OOD 引入 NOR 并扩大机制空间，有 30 个观测等价类和 train 重叠，因此度量的是算子及假设空间迁移。

## Boolean 评测口径

- 输出格式为 `Expression: <formula>`。Fixed-Prompt 文本与官方函数逐字对齐。算子按数据集 metadata 的列表顺序展示。SFT 对 AND/OR 的操作数做随机交换，并检查机制键。
- Recovery 统计匹配已存 GT 机制键的合法输出。Uniqueness 沿用官方 `novelty_rate`，分子是空间内的唯一机制键数，分母是查询次数。`n_unique_valid` 同时保留。组内奖励的距离是机制子树集合的 Jaccard 距离。
- 官方将 `Not(Or(...))` 视为 NOR，并用原始 SymPy AST 深度作空间检查。按这个规则重放原始 GT，Extended 全集有 15/510 个 GT 实例超出官方空间，同一公式落在 15 道题。ID test 为 4/124，Full OOD 为 465/975，来自 31 个公式。这些例子留在原始 GT 和查询预算里。原样重放 GT 的 Macro Recovery 为 ID test 97.71%、OOD 54.05%。统计里的 `1-(1-1/K)^K` 仍按存储的 K 计算，代表从完整 GT 均匀抽样的参考值。
- 本地 verifier 和官方 Fixed-Prompt 分支在 SymPy 前限制输入为布尔表达式语法。官方响应提取器的行为保留，例如纯字母乱码仍可能计为解析成功，随后被空间检查拒绝。
- 用 0.5B tokenizer 实测，最长目标含 `Expression:` 前缀和原始空格，ID 为 17 token，OOD 为 21 token，均未计 EOS。完整 Chat Prompt 最长分别为 550 和 560 token。SFT 最大序列长度默认 640，确保监督目标保留。
- Qwen 统一使用 temperature 0.9、top-p 0.95、最多 64 个生成 token。DeepSeek 沿用官方客户端的 temperature 0.7。离线替身验证了两份基线的采样次数、固定提示词和独立输出路径。

## 本地验证

- CPU 单元测试共 11 项通过。切分脚本打印 train/val/test 等价类零重叠。Bash dry run 校验了默认和跳过 Validity-Only 时的评测矩阵。
- Qwen2.5-0.5B-Instruct 在 RTX 4060 Laptop GPU 8 GB 上以 BF16 运行。独立 `.venv` 继承本机 PyTorch 2.6.0+cu124，使用 Transformers 4.57.6、PEFT 0.22.0 和 SymPy 1.13.1。
- 冒烟完成 SFT 4 步、Diversity 和 Validity-Only GRPO 各 2 步。两种训练均执行 Step 0 评测并写入 best 和 final。每个 ID 变体使用相同的 3 道题，每个 OOD 变体评测第一道题。
- 以下为按题平均的冒烟指标，检验训练、checkpoint 重载、评测与统计的贯通。

| 子集 | 变体 | Valid | Recovery |
|---|---|---:|---:|
| ID | Base | 0.00% | 0.00% |
| ID | SFT | 34.17% | 8.26% |
| ID | Validity-Only best | 34.17% | 8.26% |
| ID | Validity-Only final | 38.10% | 6.30% |
| ID | Diversity best | 34.17% | 8.26% |
| ID | Diversity final | 38.10% | 8.26% |
| OOD | Base | 0.00% | 0.00% |
| OOD | SFT | 80.00% | 5.00% |
| OOD | Validity-Only final | 92.50% | 5.00% |
| OOD | Diversity best | 80.00% | 5.00% |

ID 每个变体共 48 次查询，OOD 共 40 次。SFT loss 和 GRPO loss、KL 均为有限值。两个 GRPO 模式在这次短跑的验证 Recovery 都没有超过 Step 0，因此 best 保留了对应的 SFT 初始状态。

冒烟评测 JSON 已通过统计脚本并生成 Markdown、JSON 和四张图。两份 DeepSeek 离线替身输出也通过按观测 ID 对齐验证，覆盖任务书指定的六组比较。

本地输出位于 `outputs/smoke_boolean/` 和 `results/smoke_boolean/`。7B 全量结果和真实 DeepSeek 基线由运行指南中的命令生成。

## 集成时建议加入忽略规则

~~~gitignore
HypoSpace-release/boolean/grpo_lora/local_models/
HypoSpace-release/boolean/grpo_lora/outputs/smoke_*/
HypoSpace-release/boolean/grpo_lora/results/smoke_*/
~~~
