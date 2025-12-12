# -*- coding: utf-8 -*-
"""
Evaluation helpers for ICM experiments.
ICM 实验用的评估辅助脚本。

- Support five settings:
  * "zero_shot"         : base model, no context.
  * "zero_shot_chat"    : chat/Instruct model, no context.
  * "supervised"        : gold-supervision many-shot context (auto-packed).
  * "unsupervised"      : ICM-selected few-shot context.
  * "random_few_shot"   : randomly selected fixed-K few-shot, using gold labels.

- All settings use the SAME way to measure accuracy:
  parse model text output -> binary label (0/1) -> compare with gold.
  所有设置统一使用“解析模型文本输出 → 映射为 0/1 → 对比 gold label”的方式计算准确率。
"""

import os
import re
import json
import asyncio
import random
from datetime import datetime

from tqdm import tqdm

from utils.env import get_env_api_key
from utils.data import load_dataset_maybe
from utils.hyperbolic import hyperbolic_completion_with_logprobs_async
from icm.icm_main import icm_main


# ============================================================
# Helpers: model type / project root
# 辅助函数：模型类型判断 & 项目根目录
# ============================================================

def _is_chat_model(model_name: str) -> bool:
    """
    Heuristic: treat models containing 'Instruct' as chat models.
    简单启发：名字中包含 'Instruct' 的模型视为 Chat 模型。
    """
    if not model_name:
        return False
    return "Instruct" in model_name


def _resolve_project_root() -> str:
    """
    Resolve project root as parent of current file's directory.
    将项目根目录视为当前文件所在目录的上一级目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # eval/
    project_root = os.path.dirname(current_dir)  # project root
    return project_root


# ============================================================
# Token helpers (tiktoken)
# Token 估算辅助函数（使用 tiktoken）
# ============================================================

def _get_token_encoder():
    """
    Get a tiktoken encoder for approximate token counting.
    使用 tiktoken 获取一个通用编码器，用于近似计算 token 数。
    """
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "tiktoken is required for token estimation in supervised evaluation. "
            "Please install it via `pip install tiktoken`."
        ) from e

    # Use generic large context BPE (slightly different from specific models but good for budget)
    # 使用通用的大上下文 BPE（与具体模型编码略有差异，但足够用于预算控制）
    return tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str, encoder) -> int:
    """
    Count tokens of a text using the given tiktoken encoder.
    使用给定的 tiktoken encoder 统计文本的 token 数。
    """
    if not text:
        return 0
    return len(encoder.encode(text))


# ============================================================
# Prompt builders (Updated Strategy: Instruction at Top)
# 构造 Prompt（更新策略：指令置顶）
# ============================================================

def _format_example_line(ex: dict, include_answer: bool = True) -> str:
    """
    Format one example (pure pattern without instruction).
    格式化单条样本（不带指令的纯模式）。
    """
    q = (ex.get("question") or "").strip()
    c = (ex.get("choice") or "").strip()

    lines = [
        f"Question: {q}",
        f"Claim: {c}",
    ]
    if include_answer:
        # Use lowercase 'true'/'false' matching the instruction requirement.
        # 使用小写的 'true'/'false' 以匹配指令要求。
        label = ex.get("label")
        ans = "true" if label == 1 else "false"
        lines.append(f"I think this claim is {ans}")
    else:
        # For the target block (no answer yet)
        # 对于目标块（尚未回答）
        lines.append("I think this claim is")

    return "\n".join(lines)


def _build_fewshot_context(examples: list[dict]) -> str:
    """
    Build few-shot context from a list of examples (Pure examples).
    构造 Few-shot 上下文（纯示例）。
    """
    if not examples:
        return ""

    lines: list[str] = []
    for ex in examples:
        # Add answer for context examples
        # 为上下文示例添加答案
        lines.append(_format_example_line(ex, include_answer=True))
        lines.append("")  # blank line between examples

    return "\n".join(lines).strip() + "\n\n"


def _build_current_block(example: dict) -> str:
    """
    Build the prompt block for the current test example (without answer, no instruction).
    构造当前测试样本的 Prompt 段（无答案，无指令）。
    """
    return _format_example_line(example, include_answer=False)


def _build_eval_prompt(
        example: dict,
        setting: str,
        fewshot_examples: list[dict] | None = None,
) -> str:
    """
    Build the final prompt with Instruction AT THE TOP.
    构造最终 Prompt，将指令置于最顶端。
    """
    # 1. System Instruction (Global) - 仅在最上方出现一次
    # 1. 全局系统指令
    instruction = "Answer with a single word true or false.\n\n"

    # 2. Few-shot context (if any)
    # 2. Few-shot 上下文（如果有）
    context = ""
    if setting in ("supervised", "unsupervised", "random_few_shot") or setting.startswith("unsupervised_"):
        if fewshot_examples:
            context = _build_fewshot_context(fewshot_examples)

    # 3. Current Target
    # 3. 当前目标问题
    current_block = _build_current_block(example)

    # 4. Assembly
    # 4. 组装
    # Structure: [Instruction] + [Context] + [Target]
    full_prompt = instruction + context + current_block

    return full_prompt


# ============================================================
# Response parsing: text -> 0/1
# 响应解析：从模型文本输出得到 0/1 标签
# ============================================================

def _extract_label_from_response(
        resp: dict,
        is_chat: bool = False,
        debug: bool = False,
) -> int:
    """
    Extract binary label (0/1) from Hyperbolic completion/chat response.
    从 Hyperbolic completion/chat 响应中解析二元标签 0/1。
    """
    choices = resp.get("choices") or []
    if not choices:
        raise ValueError("No choices in response / 响应中没有 choices。")

    ch = choices[0]
    if is_chat:
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
    else:
        text = ch.get("text") or ""

    if debug:
        print(">>> Raw model text:")
        print(text)
        print("--------------------------------------------------")

    cleaned = (text or "").strip().lower()

    # Normalize some punctuation.
    cleaned = (
        cleaned
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("’", "'")
    )

    # Handle explicit negation
    if re.search(r"\bnot\s+true\b", cleaned) or re.search(r"\buntrue\b", cleaned):
        return 0

    # Check prefix
    for prefix, label_val in [
        ("true", 1),
        ("false", 0),
        ("yes", 1),
        ("no", 0),
    ]:
        if cleaned.startswith(prefix):
            return label_val

    # Check whole word existence
    has_true = bool(re.search(r"\btrue\b", cleaned))
    has_false = bool(re.search(r"\bfalse\b", cleaned))
    has_yes = bool(re.search(r"\byes\b", cleaned))
    has_no = bool(re.search(r"\bno\b", cleaned))

    if has_true and not has_false:
        return 1
    if has_false and not has_true:
        return 0
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0

    raise ValueError(f"Cannot parse label from model output: {text!r}")


# ============================================================
# Single call & loop over dataset
# 单条调用 & 数据集循环
# ============================================================

def _predict_one_sync(
        example: dict,
        setting: str,
        model: str,
        api_key: str,
        fewshot_examples: list[dict] | None = None,
        timeout: float = 60.0,
        top_logprobs: int = 0,
        max_tokens: int | None = None,
        debug: bool = False,
        database: bool = True,
) -> tuple[int, dict, str]:
    """
    Predict label for a single example synchronously.
    对单条样本进行同步预测，返回 (pred_label, raw_response, prompt)。
    """
    prompt = _build_eval_prompt(example, setting, fewshot_examples=fewshot_examples)
    is_chat = _is_chat_model(model)

    async def _call():
        return await hyperbolic_completion_with_logprobs_async(
            model=model,
            prompt=prompt,
            api_key=api_key,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            echo=False,
            temperature=0.0,
            top_p=1.0,
            debug=debug,
            database=database,
        )

    # Use asyncio.run with error handling
    try:
        resp = asyncio.run(_call())
    except Exception as e:
        if debug:
            print(f"[EVAL] HTTP/Async Error: {e}")
        return -1, {"error": str(e)}, prompt

    try:
        pred_label = _extract_label_from_response(resp, is_chat=is_chat, debug=debug)
        return pred_label, resp, prompt
    except Exception as e:
        if debug:
            print(f"[EVAL] Label parse error for example id={example.get('id')}: {e}")
        return -1, resp, prompt


def _predict_all_sequential(
        dataset: list[dict],
        setting: str,
        model: str,
        api_key: str,
        fewshot_examples: list[dict] | None = None,
        timeout: float = 60.0,
        top_logprobs: int = 0,
        max_tokens: int | None = None,
        debug: bool = False,
        database: bool = True,
        desc: str = "EVAL (sequential)",
) -> list[tuple[int, dict, str]]:
    """
    Predict labels for all examples in a dataset sequentially.
    以顺序方式对整个数据集逐条预测，避免过多并发导致限流。
    """
    results: list[tuple[int, dict, str]] = []
    for ex in tqdm(dataset, desc=desc):
        pred_label, raw_resp, prompt = _predict_one_sync(
            example=ex,
            setting=setting,
            model=model,
            api_key=api_key,
            fewshot_examples=fewshot_examples,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            debug=debug,
            database=database,
        )
        results.append((pred_label, raw_resp, prompt))

    return results


# ============================================================
# Evaluation main function
# 统一的评估入口
# ============================================================

def evaluate(
        data,
        setting: str,
        model: str,
        api_key: str | None = None,
        train_data_for_icm=None,
        icm_mp_method: str = "official",
        icm_alpha: float = 1.0,
        icm_target_subset_size: int = 8,
        icm_max_iter: int = 500,
        icm_consistency_mode: str = "at_most_one_true",
        icm_enforce_unique_cid: bool = False,
        icm_max_concurrent: int = 1,  # [New Param] Default to 1 for safety / 默认为 1 以确保安全
        max_context_tokens: int = 131072,
        fewshot_seed: int = 42,
        random_fewshot_k: int = 8,
        timeout: float = 60.0,
        max_tokens: int = 16,
        debug: bool = False,
        save_result: bool = True,
        result_root: str | None = None,
        icm_result_root: str | None = None,
        dataset_name: str = "truthfulqa",
):
    """
    Unified evaluation function.
    统一评估函数。
    """
    start_time = datetime.now().isoformat()

    if api_key is None:
        api_key = get_env_api_key()

    test_list = load_dataset_maybe(data)
    if not test_list:
        raise ValueError("Empty test dataset for evaluation. / 评估用测试集为空。")

    print(
        f"[EVAL] Setting={setting}, model={model}, "
        f"num_test={len(test_list)}, dataset={dataset_name}"
    )

    project_root = _resolve_project_root()
    attempt_root = None

    if result_root is not None:
        attempt_root = os.path.dirname(result_root)
        if icm_result_root is None:
            icm_result_root = os.path.join(attempt_root, "icm")
    elif save_result:
        timestamp_for_attempt = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt_root = os.path.join(project_root, "results", f"attempt_{timestamp_for_attempt}")
        result_root = os.path.join(attempt_root, "evaluation")
        if icm_result_root is None:
            icm_result_root = os.path.join(attempt_root, "icm")

    fewshot_examples: list[dict] | None = None

    # Handle settings requiring training data
    if setting in ("supervised", "unsupervised", "random_few_shot") or setting.startswith("unsupervised_"):
        if train_data_for_icm is None:
            raise ValueError(
                "train_data_for_icm is required for few-shot settings. / few-shot 设置需要提供 train_data_for_icm。"
            )

        train_list = load_dataset_maybe(train_data_for_icm)
        if not train_list:
            raise ValueError(
                "Empty training data. / 训练集为空。"
            )

        if setting == "supervised":
            encoder = _get_token_encoder()
            max_current_block_tokens = 0
            for ex in test_list:
                current_block_text = _build_current_block(ex)
                t = _count_tokens(current_block_text, encoder)
                if t > max_current_block_tokens:
                    max_current_block_tokens = t

            safety_margin = 512
            reserved_for_output = max_tokens or 0
            effective_limit = max_context_tokens
            max_input_for_context = max(
                effective_limit - reserved_for_output - max_current_block_tokens - safety_margin,
                0,
            )

            rng = random.Random(fewshot_seed)
            shuffled_train = list(train_list)
            rng.shuffle(shuffled_train)

            selected: list[dict] = []
            current_context_tokens = 0

            for ex in shuffled_train:
                # Note: include_answer=True for context construction token counting
                demo_block = _format_example_line(ex, include_answer=True) + "\n\n"
                demo_tokens = _count_tokens(demo_block, encoder)

                if current_context_tokens + demo_tokens <= max_input_for_context:
                    selected.append(ex)
                    current_context_tokens += demo_tokens
                else:
                    break

            fewshot_examples = selected
            print(
                f"[EVAL] Supervised many-shot (random+greedy): "
                f"seed={fewshot_seed}, K={len(fewshot_examples)}, "
                f"context_budget={max_input_for_context} tokens"
            )

        elif setting == "random_few_shot":
            rng = random.Random(fewshot_seed)
            shuffled_train = list(train_list)
            rng.shuffle(shuffled_train)
            k = min(max(0, random_fewshot_k), len(shuffled_train))
            fewshot_examples = shuffled_train[:k]
            print(
                f"[EVAL] Random few-shot: seed={fewshot_seed}, "
                f"K={len(fewshot_examples)} (requested={random_fewshot_k})"
            )

        elif setting == "unsupervised" or setting.startswith("unsupervised_"):
            print(f"[EVAL] Running ICM on training data for {setting} (method={icm_mp_method})...")

            # Determine sub-folder for ICM results to avoid collision between methods
            # 确定 ICM 结果的子目录，避免不同方法结果冲突
            method_specific_root = None
            if icm_result_root:
                method_specific_root = os.path.join(icm_result_root, icm_mp_method)
                os.makedirs(method_specific_root, exist_ok=True)

            icm_subset = icm_main(
                data=train_list,
                model=model,
                api_key=api_key,
                mp_method=icm_mp_method,
                alpha=icm_alpha,
                target_subset_size=icm_target_subset_size,
                max_iter=icm_max_iter,
                initial_t=10.0,
                final_t=0.01,
                decay=0.99,
                scheduler="log",
                use_consistency_term=True,
                timeout=timeout,
                top_logprobs=20,
                max_concurrent=icm_max_concurrent,  # Pass the concurrency setting / 传递并发设置
                save_result=save_result,
                result_prefix=f"icm_eval_{dataset_name}",
                seed=42,
                debug=debug,
                consistency_mode=icm_consistency_mode,
                enforce_unique_cid=icm_enforce_unique_cid,
                result_root=method_specific_root,  # Save to sub-folder / 保存到子目录
            )
            fewshot_examples = icm_subset
            print(f"[EVAL] Unsupervised ICM few-shot size={len(fewshot_examples)}")

    # Predict on test set
    seq_desc = f"EVAL (sequential, setting={setting})"
    pred_results = _predict_all_sequential(
        dataset=test_list,
        setting=setting,
        model=model,
        api_key=api_key,
        fewshot_examples=fewshot_examples,
        timeout=timeout,
        top_logprobs=0,
        max_tokens=max_tokens,
        debug=debug,
        database=True,
        desc=seq_desc,
    )

    per_example_records: list[dict] = []
    num_parsed_correct = 0
    num_parsed_incorrect = 0
    num_unparsed = 0
    num_total = len(test_list)

    for ex, (pred_label, raw_resp, prompt) in zip(test_list, pred_results):
        gold = ex.get("label")

        if pred_label == -1:
            parsed_flag = False
            correct_flag = False
            num_unparsed += 1
        else:
            parsed_flag = True
            if pred_label == gold:
                correct_flag = True
                num_parsed_correct += 1
            else:
                correct_flag = False
                num_parsed_incorrect += 1

        record = {
            "id": ex.get("id"),
            "question": ex.get("question"),
            "choice": ex.get("choice"),
            "gold_label": gold,
            "pred_label": pred_label,
            "parsed": parsed_flag,
            "correct": bool(correct_flag),
            "prompt": prompt,
            "raw_response": raw_resp,
        }
        per_example_records.append(record)

    num_parsed = num_parsed_correct + num_parsed_incorrect
    accuracy = (
        float(num_parsed_correct) / float(num_total)
        if num_total > 0 else 0.0
    )
    accuracy_on_parsed = (
        float(num_parsed_correct) / float(num_parsed)
        if num_parsed > 0 else 0.0
    )

    summary = {
        "setting": setting,
        "model": model,
        "dataset": dataset_name,
        "num_examples": num_total,
        "num_parsed": num_parsed,
        "num_parsed_correct": num_parsed_correct,
        "num_parsed_incorrect": num_parsed_incorrect,
        "num_unparsed": num_unparsed,
        "num_failed": num_unparsed,
        "accuracy": accuracy,
        "accuracy_on_parsed": accuracy_on_parsed,
        "icm_concurrency": icm_max_concurrent,
    }

    print("========== Evaluation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    result_path = None
    summary_path = None
    arguments_path = None

    if save_result and result_root is not None:
        os.makedirs(result_root, exist_ok=True)
        base_name = f"{setting}_{dataset_name}"
        result_path = os.path.join(result_root, f"{base_name}.json")
        summary_path = os.path.join(result_root, f"{base_name}_summary.json")
        arguments_path = os.path.join(result_root, f"{base_name}_arguments.json")

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(per_example_records, f, ensure_ascii=False, indent=2)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        fewshot_k = len(fewshot_examples) if fewshot_examples else 0
        eval_args = {
            "setting": setting,
            "model": model,
            "data": data if isinstance(data, str) else "<in-memory>",
            "train_data_for_icm": (
                train_data_for_icm if isinstance(train_data_for_icm, str) else "<in-memory>"
            ),
            "icm_mp_method": icm_mp_method,
            "icm_alpha": icm_alpha,
            "icm_target_subset_size": icm_target_subset_size,
            "icm_max_iter": icm_max_iter,
            "icm_consistency_mode": icm_consistency_mode,
            "icm_enforce_unique_cid": icm_enforce_unique_cid,
            "icm_max_concurrent": icm_max_concurrent,
            "max_context_tokens": max_context_tokens,
            "fewshot_seed": fewshot_seed,
            "random_fewshot_k": random_fewshot_k,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "debug": debug,
            "save_result": save_result,
            "result_root": result_root,
            "icm_result_root": icm_result_root,
            "dataset_name": dataset_name,
            "fewshot_size": fewshot_k,
            "start_time": start_time,
            "saved_time": datetime.now().isoformat(),
            "api_key_source": "env" if api_key is None else "provided_or_env",
        }

        with open(arguments_path, "w", encoding="utf-8") as f:
            json.dump(eval_args, f, ensure_ascii=False, indent=2)

        print(f"[EVAL] Per-example results saved to: {result_path}")
        print(f"[EVAL] Summary saved to: {summary_path}")
        print(f"[EVAL] Arguments saved to: {arguments_path}")

    return {
        "summary": summary,
        "per_example": per_example_records,
        "result_path": result_path,
        "summary_path": summary_path,
        "arguments_path": arguments_path,
    }


def run_evaluation_demo(settings: list[str] | None = None):
    """
    Run evaluation demo on TruthfulQA-style data.
    使用 TruthfulQA 样式数据运行评估 Demo。
    """
    project_root = _resolve_project_root()
    train_path = os.path.join(project_root, "truthfulqa_train.json")
    test_path = os.path.join(project_root, "truthfulqa_test.json")

    base_model = "meta-llama/Meta-Llama-3.1-405B"
    chat_model = "meta-llama/Meta-Llama-3.1-405B-Instruct"

    api_key = get_env_api_key()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt_root = os.path.join(project_root, "results", f"attempt_{timestamp}")
    eval_root = os.path.join(attempt_root, "evaluation")
    icm_root = os.path.join(attempt_root, "icm")
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(icm_root, exist_ok=True)

    print("========== Running Evaluation Demo on TruthfulQA ==========\n")
    print(f"[DEMO] Attempt folder: {attempt_root}")

    if settings is None:
        settings = [
            "zero_shot",
            "zero_shot_chat",
            "supervised",
            "unsupervised",
            "random_few_shot",
        ]

    seen = set()
    normalized_settings: list[str] = []
    for s in settings:
        if s not in seen:
            seen.add(s)
            normalized_settings.append(s)

    for setting in normalized_settings:
        if setting == "zero_shot":
            print("\n[DEMO] Running Zero-Shot (base model)...")
            evaluate(
                data=test_path,
                setting="zero_shot",
                model=base_model,
                api_key=api_key,
                train_data_for_icm=None,
                timeout=60.0,
                max_tokens=20,
                debug=False,
                save_result=True,
                result_root=eval_root,
                icm_result_root=icm_root,
                dataset_name="truthfulqa",
            )

        elif setting == "zero_shot_chat":
            print("\n[DEMO] Running Zero-Shot-Chat (chat model)...")
            evaluate(
                data=test_path,
                setting="zero_shot_chat",
                model=chat_model,
                api_key=api_key,
                train_data_for_icm=None,
                timeout=60.0,
                max_tokens=20,
                debug=False,
                save_result=True,
                result_root=eval_root,
                icm_result_root=icm_root,
                dataset_name="truthfulqa",
            )

        elif setting == "supervised":
            print("\n[DEMO] Running Supervised Many-Shot (base model)...")
            evaluate(
                data=test_path,
                setting="supervised",
                model=base_model,
                api_key=api_key,
                train_data_for_icm=train_path,
                icm_target_subset_size=8,
                timeout=60.0,
                max_tokens=20,
                debug=False,
                save_result=True,
                result_root=eval_root,
                icm_result_root=icm_root,
                dataset_name="truthfulqa",
            )

        elif setting == "unsupervised":
            # Loop through methods for unsupervised setting
            # 针对 unsupervised 设置循环遍历三种方法
            mp_methods = ["official", "ll_stub", "utfs"]

            # [CRITICAL] Safe concurrency default to 1 to avoid 429
            # [关键] 安全并发数设为 1 以避免 429 错误
            safe_concurrency = 1

            for mp_method in mp_methods:
                print(f"\n[DEMO] Running Unsupervised ({mp_method}, base model)...")
                evaluate(
                    data=test_path,
                    setting=f"unsupervised_{mp_method}",  # Unique setting name
                    model=base_model,
                    api_key=api_key,
                    train_data_for_icm=train_path,
                    icm_mp_method=mp_method,  # Pass current method
                    icm_alpha=1.0,
                    icm_target_subset_size=8,
                    icm_max_iter=256 * 25,
                    icm_consistency_mode="at_most_one_true",
                    icm_enforce_unique_cid=True,
                    icm_max_concurrent=safe_concurrency,  # Use safe concurrency
                    timeout=60.0,
                    max_tokens=20,
                    debug=False,
                    save_result=True,
                    result_root=eval_root,
                    icm_result_root=icm_root,
                    dataset_name="truthfulqa",
                )

        elif setting == "random_few_shot":
            print("\n[DEMO] Running Random Few-Shot (base model)...")
            evaluate(
                data=test_path,
                setting="random_few_shot",
                model=base_model,
                api_key=api_key,
                train_data_for_icm=train_path,
                random_fewshot_k=8,
                timeout=60.0,
                max_tokens=20,
                debug=False,
                save_result=True,
                result_root=eval_root,
                icm_result_root=icm_root,
                dataset_name="truthfulqa",
            )
        else:
            print(f"[DEMO] Unknown setting '{setting}', skipped. / 未知 setting '{setting}'，已跳过。")


if __name__ == "__main__":
    run_evaluation_demo()