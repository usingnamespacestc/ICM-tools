import os

def get_env_api_key() -> str:
    """
    Read Hyperbolic API key from environment variable `HYPERBOLIC_API_KEY`.
    从环境变量 `HYPERBOLIC_API_KEY` 中读取 Hyperbolic API Key。

    This is mainly used by demo helpers so we don't hard-code keys in code.
    主要用于 demo 代码，避免在源码中硬编码真实密钥。
    """
    api_key = os.environ.get("HYPERBOLIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable HYPERBOLIC_API_KEY is not set. / "
            "未设置环境变量 HYPERBOLIC_API_KEY。"
        )
    return api_key