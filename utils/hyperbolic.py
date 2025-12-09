# -*- coding: utf-8 -*-
"""
Hyperbolic API helper (async).
异步 Hyperbolic API 工具函数。

- Automatically route base models to /v1/completions (with logprobs).
- Automatically route chat / Instruct models to /v1/chat/completions (with logprobs if possible).
- 使用同一个函数自动区分 Base 与 Instruct 模型：
  * Base -> /v1/completions（支持 logprobs）
  * Instruct/Chat -> /v1/chat/completions（尝试开启 logprobs + top_logprobs）
"""

import json
import asyncio
import os
import httpx

from utils.database import (
    init_db,
    fetch_cached_result,
    save_result_to_db,
)


def _is_chat_model(model_name: str) -> bool:
    """
    Heuristic to decide whether a model should use the Chat endpoint.
    用简单启发式判断模型是否走 Chat 接口。

    For now we treat any model whose name contains "Instruct" as a chat model.
    当前策略：只要模型名中包含 "Instruct" 就视为 Chat 模型。
    """
    if not model_name:
        return False
    return "Instruct" in model_name


async def hyperbolic_completion_with_logprobs_async(
    model: str,
    prompt: str,
    api_key: str = None,
    timeout: float = 60.0,
    top_logprobs: int = 5,
    max_tokens: int = 512,
    echo: bool = False,
    temperature: float | None = None,  # Sampling temperature / 采样温度
    top_p: float | None = None,        # Nucleus sampling top_p / 核采样 top_p
    stop: list[str] | None = None,     # Optional stop tokens / 可选的停止 token 列表
    debug: bool = False,
    database: bool = True,
) -> dict | None:
    """
    Call Hyperbolic with logprobs support (async).
    异步调用 Hyperbolic 接口，并尽量开启 logprobs 功能。

    This helper auto-routes:
      - Base-like models   -> POST /v1/completions  with {"prompt": ...}
      - Chat/Instruct like -> POST /v1/chat/completions with {"messages": [...]}
    该函数会自动分流：
      - Base 模型          -> /v1/completions（使用 prompt）
      - Chat / Instruct 模型 -> /v1/chat/completions（使用 messages）

    Args / 参数:
        model:
            Model name on Hyperbolic, e.g.
            模型名称，例如：
              - "meta-llama/Meta-Llama-3.1-405B"
              - "meta-llama/Meta-Llama-3.1-405B-Instruct"

        prompt:
            Plain text prompt we build upstream (for MP / evaluation).
            上游构造的纯文本 Prompt（例如 Mutual Predictability / evaluation）。

        api_key:
            Hyperbolic API Key; if None, read from env HYPERBOLIC_API_KEY.
            Hyperbolic 的 API Key；若为 None，则从环境变量 HYPERBOLIC_API_KEY 读取。

        timeout:
            HTTP timeout in seconds.
            HTTP 超时时间（秒）。

        top_logprobs:
            How many logprobs per token to request (if supported).
            每个 token 请求多少个 logprobs（若 API 支持）。

        max_tokens:
            Max completion tokens.
            补全的最大 token 数。

        echo:
            For /v1/completions only: request echo of prompt.
            仅对 /v1/completions 有意义：是否回显 prompt。

        temperature:
            Sampling temperature for both base and chat endpoints.
            Base 与 Chat 统一使用的采样温度（None 表示用默认值）。

        top_p:
            Nucleus sampling top_p for both endpoints.
            Base 与 Chat 统一使用的核采样 top_p（None 表示用默认值）。

        stop:
            Optional list of stop strings.
            可选的停止字符串列表，作用于 Base 与 Chat。

        debug:
            Print debug info.
            是否打印调试信息。

        database:
            Whether to enable sqlite caching.
            是否启用 sqlite 级别的缓存。

    Returns / 返回:
        dict: Raw JSON response from Hyperbolic.
              Hyperbolic 返回的原始 JSON。
    """
    if debug:
        print(">>> Database caching enabled? / 是否启用数据库缓存？", database)

    # ------------------------------------------------------------
    # Initialize DB (first run will ensure table exists)
    # 显式初始化数据库（第一次运行会建库建表）
    # ------------------------------------------------------------
    if database:
        init_db()

    # ------------------------------------------------------------
    # Try database cache first
    # 优先从数据库读取缓存
    # ------------------------------------------------------------
    if database:
        try:
            cached = fetch_cached_result(
                model=model,
                prompt=prompt,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                echo=echo,
                timeout=timeout,
            )
        except Exception as e:
            if debug:
                print(f">>> Database query error, fallback to API. / 数据库查询失败，转 API：{e}")
            cached = None

        if cached is not None:
            if debug:
                print(">>> Found result in database, skipping API call. / 数据库命中缓存，跳过 API 调用。")
            return cached
        else:
            if debug:
                print(">>> No cached result found, sending API request. / 未命中缓存，发送 API 请求。")

    # ------------------------------------------------------------
    # Prepare API key
    # 准备 API Key
    # ------------------------------------------------------------
    api_key = api_key or os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key must be provided explicitly or via environment variable 'HYPERBOLIC_API_KEY'. "
            "/ 必须显式提供 API 密钥，或通过环境变量 'HYPERBOLIC_API_KEY' 提供。"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Decide endpoint & payload based on model type.
    # 根据模型类型决定调用哪个 endpoint，以及如何构造 payload。
    is_chat = _is_chat_model(model)

    if is_chat:
        # --------------------------------------------------------
        # Chat / Instruct models -> /v1/chat/completions
        # 将 prompt 包装进 messages 中，作为一条 user 消息。
        # --------------------------------------------------------
        url = "https://api.hyperbolic.xyz/v1/chat/completions"

        payload: dict = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
        }

        # Try to enable logprobs for chat if requested.
        # 按照官方文档，chat 接口使用 logprobs + top_logprobs。
        if top_logprobs and top_logprobs > 0:
            # Docs say: "logprobs must be set to true if top_logprobs is used."
            # 官方文档：若使用 top_logprobs，则 logprobs 需为 true。
            payload["logprobs"] = True
            payload["top_logprobs"] = int(top_logprobs)

        # echo is not supported in chat API -> we just ignore it.
        # Chat 接口不支持 echo 参数，这里直接忽略。

    else:
        # --------------------------------------------------------
        # Base models -> /v1/completions
        # 保持与你之前工作正常的逻辑完全兼容。
        # --------------------------------------------------------
        url = "https://api.hyperbolic.xyz/v1/completions"

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        # Only send logprobs when > 0 to keep requests clean.
        # top_logprobs > 0 时才请求 logprobs，保持请求尽量简洁。
        if top_logprobs and top_logprobs > 0:
            payload["logprobs"] = int(top_logprobs)

        if echo:
            payload["echo"] = True

    # --------------------------------------------------------
    # Unified sampling controls for BOTH base & chat
    # 统一的采样控制参数：同时作用于 base 和 chat，保证评估公平
    # --------------------------------------------------------
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if stop is not None:
        payload["stop"] = stop

    if debug:
        print(">>> Sending request to Hyperbolic API...")
        print(f"URL: {url}")
        print(f"Model: {model}")
        print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
        print("--------------------------------------------------")

    # ------------------------------------------------------------
    # HTTP request
    # 发送 HTTP 请求
    # ------------------------------------------------------------
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(f">>> HTTP status error / HTTP 状态错误: {exc}")
            if exc.response is not None:
                print("Status code / 状态码:", exc.response.status_code)
                print("Response text / 响应文本:", exc.response.text)
            raise
        except httpx.HTTPError as exc:
            print(f">>> HTTP error / HTTP 错误: {exc}")
            raise

    data = response.json()

    # ------------------------------------------------------------
    # Save to database
    # 将结果写入数据库
    # ------------------------------------------------------------
    if database:
        try:
            save_result_to_db(
                model=model,
                prompt=prompt,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                echo=echo,
                timeout=timeout,
                data=data,
            )
            if debug:
                print(">>> Result saved to database. / 结果已保存到数据库。")
        except Exception as e:
            if debug:
                print(f">>> Failed to save result to database: / 保存数据库失败: {e}")

    if debug:
        # For base completions, logprobs live in choices[0].logprobs / prompt_logprobs.
        # 对于 base completions，logprobs 一般在 choices[0].logprobs / prompt_logprobs。
        # 对于 chat，Hyperbolic（如果支持）也会在 choices[0].logprobs。
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        completion_logprobs = first.get("logprobs")
        prompt_logprobs = first.get("prompt_logprobs")
        print(">>> completion logprobs present? / 是否存在补全 logprobs？",
              completion_logprobs is not None)
        print(">>> prompt_logprobs present? / 是否存在 prompt_logprobs？",
              prompt_logprobs is not None)

    return data


# ------------------------------
# Async Example Usages / 异步示例
# ------------------------------

async def _demo_base() -> None:
    """
    Simple demo for base model with /v1/completions + logprobs.
    使用 base 模型 + /v1/completions + logprobs 的简单示例。
    """
    question = "Do more than 20% of Americans have passports?"
    choice = "No, only 10% of Americans have passports."

    base_prompt = (
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        "True or False? I think this claim is"
    )

    try:
        data = await hyperbolic_completion_with_logprobs_async(
            model="meta-llama/Meta-Llama-3.1-405B",
            prompt=base_prompt,
            timeout=120.0,
            top_logprobs=20,
            max_tokens=1,
            echo=False,
            temperature=0.0,
            top_p=1.0,
            stop=["\n"],
            debug=True,
            database=False,
        )

        print("\n--- Hyperbolic Base Completion Response ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    except RuntimeError as err:
        print(f"\nRuntime error / 运行时错误: {err}")
    except httpx.HTTPError as err:
        print(f"\nHTTP error / HTTP 错误: {err}")


async def _demo_chat() -> None:
    """
    Simple demo for Instruct model with /v1/chat/completions.
    使用 Instruct 模型 + /v1/chat/completions 的简单示例。
    （当前 chat 端 logprobs 可能为 null，因此仅作功能演示）
    """
    question = "Do more than 20% of Americans have passports?"
    choice = "No, only 10% of Americans have passports."

    prompt = (
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        "True or False? I think this claim is"
    )

    try:
        data = await hyperbolic_completion_with_logprobs_async(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            prompt=prompt,
            timeout=120.0,
            top_logprobs=5,   # Hyperbolic 目前 chat 端可能不返回 logprobs
            max_tokens=1,
            echo=True,        # ignored for chat
            temperature=0.0,
            top_p=1.0,
            stop=["\n"],
            debug=True,
            database=False,
        )

        print("\n--- Hyperbolic Chat Completion Response ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    except RuntimeError as err:
        print(f"\nRuntime error / 运行时错误: {err}")
    except httpx.HTTPError as err:
        print(f"\nHTTP error / HTTP 错误: {err}")


if __name__ == "__main__":
    # You can uncomment one of the demos below for manual testing.
    # 可按需取消注释下面任意一个 Demo 进行手动测试。

    # asyncio.run(_demo_base())
    asyncio.run(_demo_chat())
