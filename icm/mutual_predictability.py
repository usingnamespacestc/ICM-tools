# -*- coding: utf-8 -*-
"""
ICM Mutual Predictability scoring helpers.
ICM 中用于计算单条样本 Mutual Predictability 分数的辅助函数。
"""

import asyncio
import math

from utils.hyperbolic import hyperbolic_completion_with_logprobs_async
from utils.env import get_env_api_key


# ============================================================
# Basic helper functions / 基础工具函数
# ============================================================


def build_prompt_core(question: str, choice: str) -> str:
    """
    Build base prompt for (question, claim) WITHOUT system instruction.
    构造不带系统指令的 (question, choice) 基础 Prompt。

    Structure / 结构:
        Question: {question}
        Claim: {choice}
        I think this claim is
    """
    return (
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        "I think this claim is "
    )


def _format_context_examples(examples: list[dict]) -> str:
    """
    Format a list of labeled examples as context string.
    将带标签的样本列表格式化为上下文各字符串。

    Format per example / 单个样本格式:
        Question: ...
        Claim: ...
        I think this claim is true/false

    Args:
        examples: List of dicts, each must have 'question', 'choice', 'label'.
                  包含 'question', 'choice', 'label' 的字典列表。
    """
    if not examples:
        return ""

    blocks = []
    for ex in examples:
        # Convert label (0/1) to string ("false"/"true")
        # 将标签 (0/1) 转换为字符串 ("false"/"true")
        # Note: Using lowercase to match the prompt suffix style.
        # 注意：使用小写以匹配 prompt 后缀风格。
        ans = "true" if ex.get("label") == 1 else "false"

        block = build_prompt_core(ex["question"], ex["choice"]) + ans
        blocks.append(block)

    # Join with double newlines to separate examples
    # 用双换行符分隔样本
    return "\n\n".join(blocks) + "\n\n"


# ============================================================
# Mutual Predictability for single (question, choice)
# 单条 (question, choice) 的 Mutual Predictability 计算
# ============================================================


async def score_mutual_predictability_async(
        question: str,
        choice: str,
        model: str,
        api_key: str,
        context_examples: list[dict] = None,
        method: str = "official",
        timeout: float = 60.0,
        top_logprobs: int = 20,
        debug: bool = False,
        database: bool = True,
) -> dict:
    """
    Compute a Mutual Predictability-style score for (question, choice).
    为 (question, choice) 计算 Mutual Predictability 风格的分数。

    Now supports few-shot context via `context_examples`.
    现在通过 `context_examples` 支持 few-shot 上下文。

    Args / 参数:
        question: Question text.
        choice: Claim text.
        model: Model name.
        api_key: API key.
        context_examples: List of other labeled examples to serve as context (D \ {x_i}).
        method: "official", "utfs", or "ll_stub".
        timeout: Request timeout.
        top_logprobs: Number of logprobs to request.
        debug: Debug mode.
        database: Use caching.
    """
    if method not in {"official", "ll_stub", "utfs"}:
        raise ValueError(
            'method must be "official", "utfs" or "ll_stub".'
        )

    # 1. Build context string if provided.
    # 1. 如果提供了上下文示例，则构建上下文字符串。
    context_str = _format_context_examples(context_examples) if context_examples else ""

    # 2. Build the current example prompt core.
    # 2. 构建当前样本的核心 Prompt。
    current_prompt = build_prompt_core(question, choice)

    # 3. Concatenate: [Context] + [Current].
    # 3. 拼接：[上下文] + [当前样本]。
    full_prompt = context_str + current_prompt

    # --------------------------------------------------------------
    # Method: "official" or "utfs" (original first-token heuristic)
    # 方法 "official" 或 "utfs"：官方 first-token 启发式
    # --------------------------------------------------------------
    if method == "official" or method == "utfs":
        response = await hyperbolic_completion_with_logprobs_async(
            model=model,
            prompt=full_prompt,  # Use full prompt with context / 使用带上下文的完整 Prompt
            api_key=api_key,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_tokens=1,
            echo=False,
            temperature=0.0,
            top_p=1.0,
            debug=debug,
            database=database,
        )

        choices = response.get("choices") or []
        if not choices:
            raise ValueError("No choices in response. / 响应中没有 choices 字段。")

        choice_obj = choices[0]
        logprobs_block = choice_obj.get("logprobs")
        if not logprobs_block:
            raise ValueError("No logprobs in response. / 响应中没有 logprobs 字段。")

        top_list = logprobs_block.get("top_logprobs")
        if not top_list:
            raise ValueError(
                "No top_logprobs in logprobs. / logprobs 中没有 top_logprobs 字段。"
            )

        first_dist = top_list[0]
        eps = 1e-5
        prob_true = eps
        prob_false = eps

        for token_text, logp in first_dist.items():
            lower = token_text.lower()
            has_true = "true" in lower
            has_false = "false" in lower

            if has_true == has_false:
                continue

            if has_true:
                prob_true += math.exp(float(logp))
            else:
                prob_false += math.exp(float(logp))

        if prob_true == eps and prob_false == eps:
            score = 0.0
        else:
            score = math.log(prob_true) - math.log(prob_false)

        return {
            "score": score,
            "method": method,
            "question": question,
            "choice": choice,
            "raw": response,
        }

    # --------------------------------------------------------------
    # Method: "ll_stub" (echo + stub log-likelihood comparison)
    # 方法 "ll_stub"：echo + stub 对数似然比较
    # Now enhanced to check both lowercase and uppercase variants.
    # 现在增强为同时检查小写和大写变体。
    # --------------------------------------------------------------
    elif method == "ll_stub":
        base_len = len(full_prompt)

        async def _score_stub(label_stub: str) -> float:
            """
            Score a specific stub string.
            为特定的 stub 字符串打分。
            """
            prompt_with_stub = full_prompt + label_stub

            data = await hyperbolic_completion_with_logprobs_async(
                model=model,
                prompt=prompt_with_stub,
                api_key=api_key,
                timeout=timeout,
                top_logprobs=top_logprobs,
                max_tokens=0,
                echo=True,
                temperature=0.0,
                top_p=1.0,
                debug=debug,
                database=database,
            )

            choices_inner = data.get("choices") or []
            if not choices_inner:
                raise ValueError("No choices in stub response.")

            choice_inner = choices_inner[0]
            logprobs_inner = choice_inner.get("logprobs")
            if not logprobs_inner:
                raise ValueError("No logprobs in stub response.")

            offsets = logprobs_inner.get("text_offset")
            token_logprobs = logprobs_inner.get("token_logprobs")

            if offsets is None or token_logprobs is None:
                raise ValueError("Missing offsets or logprobs.")

            total = 0.0
            for offset, _logp in zip(offsets, token_logprobs):
                # Only sum tokens strictly after the base prompt (the stub part)
                # 只累加 base prompt 之后的部分（即 stub 部分）
                if offset >= base_len:
                    total += float(_logp)

            return total

        # Define candidates to check. Without instruction, model might output " True" or " true".
        # 定义要检查的候选词。在没有指令的情况下，模型可能会输出 " True" 或 " true"。
        candidates_true = [" true", " True"]
        candidates_false = [" false", " False"]

        # Collect tasks
        tasks = []
        for t in candidates_true:
            tasks.append(_score_stub(t))
        for f in candidates_false:
            tasks.append(_score_stub(f))

        # Run all stub requests concurrently
        # 并发运行所有 stub 请求
        results = await asyncio.gather(*tasks)

        # Split results
        mid = len(candidates_true)
        probs_true_list = results[:mid]
        probs_false_list = results[mid:]

        # Helper: LogSumExp to aggregate probabilities in log space
        # 辅助函数：使用 LogSumExp 在对数空间聚合概率
        # P(True_Class) = P(" true") + P(" True")
        # log(P) = log( exp(logp1) + exp(logp2) )
        def log_sum_exp(log_probs):
            if not log_probs:
                return -float('inf')
            m = max(log_probs)
            if m == -float('inf'):
                return -float('inf')
            return m + math.log(sum(math.exp(lp - m) for lp in log_probs))

        total_logp_true = log_sum_exp(probs_true_list)
        total_logp_false = log_sum_exp(probs_false_list)

        score_b = total_logp_true - total_logp_false

        return {
            "score": score_b,
            "method": "ll_stub",
            "question": question,
            "choice": choice,
            "logp_true": total_logp_true,
            "logp_false": total_logp_false,
        }

    return {}


async def ab_demo() -> None:
    """
    Minimal demo for both methods "official" and "ll_stub".
    同时演示 "official" 与 "ll_stub" 两种方法的最小示例。
    """
    question = "Do more than 20% of Americans have passports?"
    choice = "No, only 10% of Americans have passports."

    model = "meta-llama/Meta-Llama-3.1-405B"
    api_key = get_env_api_key()

    print("=== Method: official ===")
    result_official = await score_mutual_predictability_async(
        question=question,
        choice=choice,
        model=model,
        api_key=api_key,
        method="official",
        timeout=120.0,
        top_logprobs=20,
        debug=True,
        database=False,
    )
    print("Score (official):", result_official["score"])

    print("\n=== Method: ll_stub ===")
    result_ll = await score_mutual_predictability_async(
        question=question,
        choice=choice,
        model=model,
        api_key=api_key,
        method="ll_stub",
        timeout=120.0,
        top_logprobs=20,
        debug=True,
        database=False,
    )
    print("Score (ll_stub):", result_ll["score"])
    print("logP(true) :", result_ll["logp_true"])
    print("logP(false):", result_ll["logp_false"])


if __name__ == "__main__":
    asyncio.run(ab_demo())