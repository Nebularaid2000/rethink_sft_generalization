# Evaluation guide

LLM evaluation suite that integrates many benchmark datasets, covering instruction following, knowledge reasoning, hallucination detection, code generation, mathematical reasoning, open-ended QA, and safety red teaming.

## Directory Structure

```text
├── scripts/                          # Evaluation scripts (entry points)
├── data/                             # Evaluation data (local)
├── results/                          # Evaluation outputs
│   ├── lm_eval/
│   ├── evalchemy/
│   ├── math/
│   ├── alpaca/
│   └── harmbench/
├── lm-evaluation-harness/            # EleutherAI lm-eval-harness framework
│   └── nltk_data/                    #   Optional local NLTK resources for IFEval (not tracked on GitHub)
├── evalchemy/                        # Evalchemy framework
├── math_eval/                        # Math evaluation framework
├── harmbench/                        # HarmBench framework (HEx-PHI safety benchmark)
└── alpaca_eval/                      # AlpacaEval framework
    ├── code/                         #   generate_alpaca.py, evaluate_reward.py
    └── alpac_eval/                   #   AlpacaEval library
```

## Benchmark Overview

| Script | Benchmark | Framework | Dimension |
|------|--------|------|------|
| `run_lm_eval.sh` | IFEval | lm-evaluation-harness | Instruction following |
| `run_lm_eval.sh` | MMLU-Pro | lm-evaluation-harness | Knowledge reasoning |
| `run_lm_eval.sh` | TruthfulQA | lm-evaluation-harness | Truthfulness |
| `run_lm_eval.sh` | HaluEval | lm-evaluation-harness | Hallucination detection |
| `run_evalchemy.sh` | GPQA Diamond | evalchemy | Graduate-level reasoning |
| `run_evalchemy.sh` | LiveCodeBench v2 | evalchemy | Code generation |
| `run_math.sh` | MATH-500 | math_eval | Mathematical reasoning |
| `run_math.sh` | AIME24 | math_eval | Competition mathematics |
| `run_alpaca.sh` | AlpacaEval | alpaca_eval + reward model | Open-ended QA |
| `run_hex_phi_generate.sh` | HEx-PHI (generation) | harmbench | Safety / jailbreak response generation |
| `run_hex_phi_judge.sh` | HEx-PHI (judge) | harmbench | Safety / ASR evaluation |

## Environment Requirements

Aside from the requirements mentioned in the main README file, you need to additionally install lm-evaluation-harness in editable mode.

```bash
cd /path/to/sft_generalization/evaluation

# install lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```


## Usage

### 1. Set Model Paths

For `run_lm_eval.sh`, `run_evalchemy.sh`, `run_math.sh`, and `run_alpaca.sh`, fill in model paths in the `models=()` array:

```bash
models=(
    /path/to/model_checkpoint_1
    /path/to/model_checkpoint_2
)
```

For HEx-PHI (`run_hex_phi_generate.sh` / `run_hex_phi_judge.sh`), configure model aliases in `harmbench/configs/model_configs/models.yaml`, then fill alias names in the script arrays:

```bash
models=(
    model_alias1
    model_alias2
)
```

### 2. Run Evaluations

```bash
# ifeval, mmlu_pro, truthfulqa, halueval
bash scripts/run_lm_eval.sh

# gpqa, livecodebench
bash scripts/run_evalchemy.sh

# math500, aime24
bash scripts/run_math.sh

# alpaca (generate + reward evaluate)
bash scripts/run_alpaca.sh

# HEx-PHI (harmbench): generate
bash scripts/run_hex_phi_generate.sh

# judge step uses OpenAI API by default (required)
export OPENAI_API_KEY=<your_api_key>

# optional: for OpenAI-compatible endpoints/proxies
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

# HEx-PHI (harmbench): judge + ASR
bash scripts/run_hex_phi_judge.sh
```

### 3. View Results

Results are written to corresponding subdirectories under `results/`.

## Data Locations

Most benchmark datasets are localized. Large assets that are excluded from GitHub are listed below.

| Benchmark | Data Path | Format |
|--------|---------|------|
| IFEval | `data/ifeval/train.jsonl` | jsonl |
| MMLU-Pro | `data/mmlu_pro/` | parquet |
| TruthfulQA | `data/truthfulqa/generation/` | HuggingFace dataset |
| HaluEval | `data/halueval/{qa,dialogue,summarization}_samples/` | HuggingFace dataset |
| GPQA Diamond | `evalchemy/eval/chat_benchmarks/GPQADiamond/data/gpqa_diamond.csv` | csv (framework-provided) |
| LiveCodeBench v2 | `data/livecodebench/` | HuggingFace dataset repo mirror (prepared locally, not tracked on GitHub) |
| MATH-500 | `data/math/MATH500/test.jsonl` | jsonl |
| AIME24 | `data/math/converted_aime_dataset/` | HuggingFace dataset (Arrow format) |
| AlpacaEval | `data/alpaca/alpaca_eval.json` | json |
| HEx-PHI | `harmbench/data/behavior_datasets/hex-phi.csv` | csv |

### Large Assets Not Tracked on GitHub

- `data/livecodebench/` is not uploaded to GitHub due to dataset size.  
  Prepare it from: https://huggingface.co/datasets/livecodebench/code_generation_lite  


- `lm-evaluation-harness/nltk_data/` is also not uploaded due to size. IFEval may require NLTK tokenizer resources from: https://github.com/nltk/nltk_data

## Configuration

Key configurable parameters in each script:

| Parameter | Description | Default |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0,1` |
| `gpu_num` / `NUM_GPUS` | Tensor parallel size | `2` |
| `default_max_model_tokens` | Maximum model context length | `32768` |
| `batch_size` | Inference batch size | `auto` |
| `tasks` / `datasets` | Benchmark task list | See each script |

## Code Sources

| Framework | Source |
|------|------|
| lm-evaluation-harness | [Transferability-of-LLM-Reasoning](https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning) `eval/lm-evaluation-harness` |
| evalchemy | [Transferability-of-LLM-Reasoning](https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning) `eval/evalchemy` |
| alpaca_eval | [LLM-Extrapolation](https://github.com/chujiezheng/LLM-Extrapolation) `code/` + `alpac_eval/` |
| harmbench | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
