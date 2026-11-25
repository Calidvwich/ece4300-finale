# HypoSpace

HypoSpace evaluates how well an LLM can produce **valid, distinct, and broad sets of hypotheses** when observations allow more than one answer. It includes causal graph inference, gravity-constrained 3D voxel reconstruction, and Boolean genetic interaction tasks.

Paper: [HypoSpace, ICML 2026 / PMLR](https://proceedings.mlr.press/v306/chen26bz.html) · [Code repository](https://github.com/CTT-Pavilion/_HypoSpace)

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended (tested in the project environment with Python 3.12)
- Internet access only for API-backed benchmark runs
- A compatible API key for the selected provider. Dataset generation and offline evaluation do not need a key.

The dependency list is in `requirements.txt`. The benchmark reports Validity, Uniqueness, and Recovery, with per-task outputs, token usage, and checkpoints.

## 2. Install after extracting the ZIP

Open PowerShell in the extracted `HypoSpace` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, keep the environment inactive and call its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, create `.venv` with `python3 -m venv .venv`, activate using `source .venv/bin/activate`, then run `pip install -r requirements.txt`.

## 3. Configure API credentials

The provided DeepSeek configs use the API model name `deepseek-flash` and read the key from `HYPOSPACE_DEEPSEEK_API_KEY`. Set it for the current PowerShell window (recommended; do not put secrets in YAML or source files):

```powershell
$env:HYPOSPACE_DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
```

To save it for future Windows sessions instead:

```powershell
[Environment]::SetEnvironmentVariable("HYPOSPACE_DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY", "User")
```

After setting a persistent variable, open a new terminal. The API key is never included in the release archive. DeepSeek config files are `causal/config/config_deepseek.yaml`, `3d/config/config_deepseek.yaml`, and `boolean/config/config_deepseek.yaml`.

The runners also support OpenAI, Anthropic, and OpenRouter via their respective environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) and config provider/model fields.

## 4. Datasets

The benchmark runners consume JSON datasets. Generate them from the project root as follows. Output folders are created automatically.

### Causal graphs

Generate the small exhaustive three-node dataset:

```powershell
python causal/generate_causal_dataset.py --nodes 3 --seed 33550336 --output causal/datasets/n3_all_observations.json
```

Generate a larger sampled-observation dataset with complete ground-truth enumeration inside its edge constraint:

```powershell
python causal/generate_causal_dataset_for_large.py --nodes 5 --n-observations 5 --fixed --n-observation-sets 80 --seed 33550336 --output causal/datasets/n5_obs5_hybrid80.json
```

The checked-in project workspace also contains `causal/datasets/n4_obs3_all.json` (392 tasks, all four-node DAG hypotheses for each task) and other experiment datasets. Their JSON metadata and `ground_truth_graphs` define each task’s exact admissible set.

### 3D reconstruction

```powershell
python 3d/generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks 1 --fixed --seed 33550336 --output 3d/datasets/grid3_h3_b1.json
```

The included generator currently supports one top-view observation per task. The paper also discusses multi-view settings; those settings are not implemented by this checkout.

### Boolean interactions

```powershell
python boolean/boolean_dataset.py --operators basic --max-depth 2 --seed 33550336 --output boolean/datasets/basic_d2.json
```

`--operators` accepts `basic`, `extended`, or `full`; `--max-depth` controls expression complexity. The generator enumerates all observation sets for the requested configuration and stores compatible expressions in the dataset.

### Included benchmark datasets

Generated datasets used by previous project experiments are under `causal/datasets/`, `causal/benchmark_obs_count/`, and `causal/benchmark_obs_count/paper_scale/`. They are JSON files and can be run directly; regenerating them is optional. Avoid overwriting these files if you want to preserve the recorded experiment state.

## 5. Run a benchmark

Run from the project root so relative paths resolve as shown.

### Causal

```powershell
python causal/run_causal_benchmark.py --dataset causal/datasets/n3_all_observations.json --config causal/config/config_deepseek.yaml --n-samples 30 --query-multiplier 1.0 --seed 33550336 --output results/causal_n3_deepseek.json
```

For the checked-in four-node/three-observation dataset, a complete paper-style per-task query budget (`N=|H_O|`) is:

```powershell
python causal/run_causal_benchmark.py --dataset causal/datasets/n4_obs3_all.json --config causal/config/config_deepseek.yaml --n-samples 392 --query-multiplier 1.0 --independent-samples --seed 33550336 --checkpoint-dir checkpoints/causal_n4 --output results/causal_n4_deepseek.json
```

Omit `--independent-samples` to use the clone's default behavior, which includes earlier proposals in later prompts for the same task. These are distinct sampling protocols; report them separately. `--query-multiplier 1.0` sets the number of calls per task to its ground-truth count. Use smaller values for smoke/low-cost checks.

### 3D

```powershell
python 3d/run_3d_benchmark.py --dataset 3d/datasets/grid3_h3_b1.json --config 3d/config/config_deepseek.yaml --n-samples 9 --query-multiplier 1.0 --seed 33550336 --output results/3d_deepseek.json
```

### Boolean

```powershell
python boolean/boolean_benchmark.py --dataset boolean/datasets/basic_d2.json --config boolean/config/config_deepseek.yaml --n-samples 35 --query-multiplier 1.0 --seed 33550336 --output results/boolean_deepseek.json
```

The number of tasks available in each dataset is shown in its `metadata` section. To evaluate every task, set `--n-samples` to `metadata.total_observation_sets` (or the total number of datasets in that file). Full `N=|H_O|` evaluations may require many API calls; check dataset ground-truth counts before running.

## 6. Outputs and resume behavior

- Benchmark JSON results are written to the `--output` path.
- Intermediate checkpoints go to `--checkpoint-dir` (or the config default where supported).
- Use a new output/checkpoint directory for each run to keep experiments separate.
- Each result records task-level metrics, generated hypotheses, token counts, costs as estimated by the local adapter, and errors.
- The local adapter’s reported cost may not equal the provider invoice. Confirm actual charges in the API provider dashboard.

## 7. Repository layout

```text
3d/       3D dataset generator, benchmark runner, models and LLM adapter
boolean/  Boolean dataset generator, benchmark runner, models and LLM adapter
causal/   Causal datasets, generators, benchmark runner and experiments
figs/     Paper and project figures
results/  Saved benchmark outputs
checkpoints/ Saved run checkpoints
```

## Citation

```bibtex
@InProceedings{pmlr-v306-chen26bz,
  title = {{H}ypo{S}pace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds},
  author = {Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Goyal, Anirudh and Ong, Yew-Soon and Liu, Dianbo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages = {15451--15465},
  year = {2026},
  volume = {306},
  series = {Proceedings of Machine Learning Research}
}
```
