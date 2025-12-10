# ICM-tools CLI Usage Guide (`main.py`)

This README explains how to use the top-level CLI script `main.py` for running evaluations and ICM searches in the ICM-tools project.

---

## 1. Overview

`main.py` allows you to:

- Run multiple evaluation settings in a single attempt.
- Optionally run a standalone ICM search on the training dataset.
- Store all outputs under a single attempt folder:

```
results/
  attempt_{YYYYMMDD_HHMMSS}/
    evaluation/
      ... per-setting JSON outputs
      ... per-setting *_arguments.json
    icm/
      target_subset.json
      ud.json
      icm_arguments.json
    output.txt   # full console log
```

Each run of `python main.py ...` creates one new attempt folder.

---

## 2. Requirements

### Python Environment

- Python 3.10+ recommended
- Install dependencies:

```
pip install -r requirements.txt
```

### Directory Structure (Simplified)

```
ICM-tools/
  main.py
  truthfulqa_train.json
  truthfulqa_test.json
  eval/
    evaluation.py
  icm/
    icm_main.py
  utils/
    env.py
  results/
```

### API Key

`main.py` automatically loads your API key via:

```python
from utils.env import get_env_api_key
```

Ensure your environment variable is set accordingly.

---

## 3. Basic Usage

### Run all default evaluation settings

```
python main.py
```

This runs:

- zero_shot
- zero_shot_chat
- supervised
- unsupervised
- random_few_shot

All outputs go to:

```
results/attempt_YYYYMMDD_HHMMSS/
```

---

## 4. Evaluation Options

### Disable evaluation

```
python main.py --no-eval
```

### Specify evaluation settings

```
python main.py --eval-settings zero_shot,unsupervised
```

### Use custom dataset paths

```
python main.py --train-path path/to/train.json --test-path path/to/test.json
```

### Select models

```
--base-model meta-llama/Meta-Llama-3.1-405B
--chat-model meta-llama/Meta-Llama-3.1-405B-Instruct
```

### Generation parameters

```
--timeout 60
--max-tokens 20
--debug
```

### Random few-shot K

```
--random-fewshot-k 8
```

---

## 5. Running ICM

### Enable standalone ICM

```
python main.py --do-icm
```

ICM results go into:

```
results/attempt_xxxx/icm/
```

### Main ICM Parameters

```
--icm-mp-method official|ll_stub
--icm-alpha 1.0
--icm-target-subset-size 8
--icm-max-iter 6400
--icm-consistency-mode at_most_one_true|conflict_count
--icm-enforce-unique-cid
--icm-initial-t 5.0
--icm-final-t 0.1
--icm-decay 0.98
--icm-scheduler log|exp
```

---

## 6. Output Files

### `evaluation/`

- `{setting}.json`
- `{setting}_arguments.json`

### `icm/`

- `target_subset.json`
- `ud.json`
- `icm_arguments.json`

### Root

- `output.txt` — full console log

---

## 7. Example Commands

Run zero-shot + unsupervised:

```
python main.py --eval-settings zero_shot,unsupervised
```

Run evaluation + standalone ICM:

```
python main.py --do-icm
```

Use a different model:

```
python main.py --base-model your/model
```

---

## 8. Notes

- All paths are resolved relative to `main.py` location.
- Each attempt is isolated and reproducible.
- A tee mechanism copies all logs to `output.txt`.

---

## 9. Support

Check `output.txt` for detailed logs if errors occur.
