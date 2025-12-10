# ICM-tools 命令行使用说明 (`main.py`)

本说明介绍如何使用 ICM-tools 项目的顶层 CLI 脚本 `main.py`，用于执行评估与 ICM 搜索。

---

## 1. 功能概述

`main.py` 可以：

- 在一次 attempt 中运行多个评估设置
- 可选地在训练集上运行一次独立 ICM 搜索
- 将所有输出统一保存在一个时间戳目录：

```
results/
  attempt_{YYYYMMDD_HHMMSS}/
    evaluation/
      ... 各 setting 的 JSON
      ... 各 setting 的 *_arguments.json
    icm/
      target_subset.json
      ud.json
      icm_arguments.json
    output.txt   # 全部日志输出
```

每次运行命令都会生成一个新的 attempt 文件夹。

---

## 2. 环境需求

### Python 环境

- 推荐 Python 3.10+
- 安装依赖：

```
pip install -r requirements.txt
```

### 项目结构（简化）

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

自动从环境变量加载：

```python
from utils.env import get_env_api_key
```

确保环境变量正确设置。

---

## 3. 基本用法

### 运行默认评估设置

```
python main.py
```

默认包含：

- zero_shot
- zero_shot_chat
- supervised
- unsupervised
- random_few_shot

输出目录：

```
results/attempt_YYYYMMDD_HHMMSS/
```

---

## 4. 评估选项

### 禁用评估

```
python main.py --no-eval
```

### 指定评估设置

```
python main.py --eval-settings zero_shot,unsupervised
```

### 自定义数据路径

```
python main.py --train-path 数据.json --test-path 数据.json
```

### 指定模型

```
--base-model meta-llama/Meta-Llama-3.1-405B
--chat-model meta-llama/Meta-Llama-3.1-405B-Instruct
```

### 调整生成参数

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

## 5. ICM 搜索

### 启动独立 ICM 搜索

```
python main.py --do-icm
```

输出目录：

```
results/attempt_xxxx/icm/
```

### 常用 ICM 参数

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

## 6. 输出说明

### evaluation/

- `{setting}.json`
- `{setting}_arguments.json`

### icm/

- `target_subset.json`
- `ud.json`
- `icm_arguments.json`

### 根目录

- `output.txt` — 完整日志

---

## 7. 示例命令

只运行 zero_shot + unsupervised：

```
python main.py --eval-settings zero_shot,unsupervised
```

运行评估 + ICM：

```
python main.py --do-icm
```

更换模型：

```
python main.py --base-model your/model
```

---

## 8. 注意事项

- 所有路径均相对于 `main.py` 所在目录解析。
- 每次运行生成独立 attempt，便于复现与对比。
- tee 机制会将所有输出写入 output.txt。

---

## 9. 故障排查

遇到问题请查看 attempt 目录内的 `output.txt`。
