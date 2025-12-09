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


def build_prompt(question: str, choice: str) -> str:
    """
    Build base prompt for (question, claim).
    为 (question, choice) 构造基础 Prompt。

    Template / 模板:
        Answer with a single word true or false.
        Question: {question}
        Claim: {choice}
        I think this claim is
    """
    return (
        "Answer with a single word true or false.\n"
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        "I think this claim is "
    )


# ============================================================
# Mutual Predictability for single (question, choice)
# 单条 (question, choice) 的 Mutual Predictability 计算
# ============================================================


async def score_mutual_predictability_async(
    question: str,
    choice: str,
    model: str,
    api_key: str,
    method: str = "official",
    timeout: float = 60.0,
    top_logprobs: int = 20,
    debug: bool = False,
    database: bool = True,
) -> dict:
    """
    Compute a Mutual Predictability-style score for (question, choice).
    为 (question, choice) 计算 Mutual Predictability 风格的分数。

    Supported methods / 支持的方法:

    - "official":
        Use the original heuristic: generate 1 token after
        "I think this claim is", and look at the first-token top_logprobs
        mass on tokens containing "true" vs "false".
        使用官方启发式：在 "I think this claim is" 之后生成 1 个 token，
        在第一个生成 token 的 top_logprobs 中，累加包含 "true" 与
        "false" 的 token 概率，比较两者的对数差。

    - "ll_stub":
        Use the echo+stub trick: compare log P(prompt + " true")
        vs log P(prompt + " false") under the language model.
        使用 echo+stub 技巧：分别计算 log P(prompt + " true")
        和 log P(prompt + " false")，返回两者差值。

    Args / 参数:
        question (str):
            Question text from dataset.
            来自数据集的 question 文本。

        choice (str):
            Claim or candidate answer to be judged.
            需要判断真伪的陈述（choice 候选答案）。

        model (str):
            Hyperbolic model name, e.g. "meta-llama/Meta-Llama-3.1-405B".
            Hyperbolic 模型名称，例如 "meta-llama/Meta-Llama-3.1-405B"。

        api_key (str):
            Hyperbolic API key.
            Hyperbolic 的 API 密钥。

        method (str):
            "official" -> original first-token heuristic.
            "official" 表示官方 first-token 启发式。

            "ll_stub"  -> echo + stub log-likelihood comparison.
            "ll_stub" 表示 echo + stub 对数似然比较方法。

        timeout (float):
            Request timeout in seconds.
            请求超时时间（秒）。

        top_logprobs (int):
            Number of top-k logprobs per token.
            每个 token 返回的前 k 个 logprob。

        debug (bool):
            Whether to print debug information.
            是否打印调试信息。

        database (bool):
            Whether to enable sqlite-based caching inside the Hyperbolic helper.
            是否启用 sqlite 缓存（在 Hyperbolic 封装内部使用）。

    Returns / 返回:
        dict:
            Contains the score and extra context, e.g.：
            返回一个包含分数和上下文信息的字典，例如：

            {
                "score": float,          # log P(true) - log P(false)
                "method": "official" or "ll_stub",
                "question": str,
                "choice": str,
                ... extra fields ...
            }
    """
    if method not in {"official", "ll_stub"}:
        raise ValueError(
            'method must be "official" or "ll_stub". / method 参数必须是 "official" 或 "ll_stub"。'
        )

    # --------------------------------------------------------------
    # Method: "official" (original first-token heuristic)
    # 方法 "official"：官方 first-token 启发式
    # --------------------------------------------------------------
    if method == "official":
        # 1. Build the prompt with "Question / Claim / I think this claim is".
        #    使用固定模板构造 Prompt。
        # 2. Set max_tokens=1 so the model only generates one token.
        #    设置 max_tokens=1，只看第一个生成 token 的分布。
        # 3. From the first token's top_logprobs, accumulate probabilities
        #    for tokens containing "true" vs "false".
        #    在第一个 token 的 top_logprobs 中累加包含 "true"/"false" 的 token 概率。
        # 4. Return log P(true) - log P(false).
        #    返回 log P(true) - log P(false) 作为分数。
        prompt = build_prompt(question, choice)

        response = await hyperbolic_completion_with_logprobs_async(
            model=model,
            prompt=prompt,
            api_key=api_key,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_tokens=1,
            echo=False,
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

        # Only look at the first generated token’s candidates.
        # 只看第一个生成 token 的候选分布。
        first_dist = top_list[0]

        eps = 1e-5
        prob_true = eps
        prob_false = eps

        # Accumulate probability mass for tokens containing "true"/"false".
        # 对包含 "true"/"false" 的 token 进行概率累加。
        for token_text, logp in first_dist.items():
            lower = token_text.lower()
            has_true = "true" in lower
            has_false = "false" in lower

            # If both or neither appear, the token is ambiguous; skip it.
            # 若同时包含或都不包含 true/false，则视为模糊，跳过。
            if has_true == has_false:
                continue

            if has_true:
                prob_true += math.exp(float(logp))
            else:
                prob_false += math.exp(float(logp))

        # If neither side receives any mass, treat as neutral score 0.
        # 若两侧都未累积到概率，则视为中性分数 0。
        if prob_true == eps and prob_false == eps:
            score = 0.0
        else:
            score = math.log(prob_true) - math.log(prob_false)

        return {
            "score": score,
            "method": "official",
            "question": question,
            "choice": choice,
            "raw": response,
        }

    # --------------------------------------------------------------
    # Method: "ll_stub" (echo + stub log-likelihood comparison)
    # 方法 "ll_stub"：echo + stub 对数似然比较
    # --------------------------------------------------------------
    base_prompt = build_prompt(question, choice)
    base_len = len(base_prompt)

    async def _score_stub(label_stub: str) -> float:
        """
        Score one label stub using echo=True, max_tokens=0.
        使用 echo=True, max_tokens=0 计算一个标签 stub 的对数概率。
        """
        full_prompt = base_prompt + label_stub

        data = await hyperbolic_completion_with_logprobs_async(
            model=model,
            prompt=full_prompt,
            api_key=api_key,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_tokens=0,
            echo=True,
            debug=debug,
            database=database,
        )

        choices_inner = data.get("choices") or []
        if not choices_inner:
            raise ValueError(
                "No choices in stub response. / stub 响应中没有 choices。"
            )

        choice_inner = choices_inner[0]
        logprobs_inner = choice_inner.get("logprobs")
        if not logprobs_inner:
            raise ValueError(
                "No logprobs in stub response. / stub 响应中没有 logprobs。"
            )

        offsets = logprobs_inner.get("text_offset")
        token_logprobs = logprobs_inner.get("token_logprobs")

        if offsets is None or token_logprobs is None:
            raise ValueError(
                "Missing text_offset or token_logprobs in stub response. / "
                "stub 响应中缺少 text_offset 或 token_logprobs。"
            )

        if len(offsets) != len(token_logprobs):
            raise ValueError(
                "Length mismatch between text_offset and token_logprobs. / "
                "text_offset 与 token_logprobs 长度不一致。"
            )

        total = 0.0
        # Only sum tokens whose text_offset >= base_len (the stub part).
        # 只累加 offset >= base_len 的 token（也就是 stub 部分）。
        for offset, _logp in zip(offsets, token_logprobs):
            if offset >= base_len:
                total += float(_logp)

        return total

    # Compute log probabilities for " true" and " false" in parallel.
    # 并行计算 " true" 和 " false" 两个 stub 的对数概率。
    logp_true, logp_false = await asyncio.gather(
        _score_stub(" true"),
        _score_stub(" false"),
    )

    score_b = logp_true - logp_false

    return {
        "score": score_b,
        "method": "ll_stub",
        "question": question,
        "choice": choice,
        "logp_true": logp_true,
        "logp_false": logp_false,
    }


# ============================================================
# Simple AB demo: test single example
# 简单 AB Demo：在单条样本上测试两种方法
# ============================================================


async def ab_demo() -> None:
    """
    Minimal demo for both methods "official" and "ll_stub".
    同时演示 "official" 与 "ll_stub" 两种方法的最小示例。
    """
    question = "Do more than 20% of Americans have passports?"
    choice = "No, only 10% of Americans have passports."

    model = "meta-llama/Meta-Llama-3.1-405B"
    api_key = get_env_api_key()  # Read from env / 从环境变量读取

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
    # Only run AB demo here.
    # 在本文件中只跑 AB demo，方便单独测试 Mutual Predictability。
    asyncio.run(ab_demo())
