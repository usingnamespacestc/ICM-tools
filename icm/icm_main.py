# -*- coding: utf-8 -*-
"""
ICM top-level helpers: compute U(D) using mutual predictability
and (optionally) a logical consistency term, plus a simulated-annealing
search over subsets / labels.

ICM 顶层辅助函数：
- 基于 Mutual Predictability 与（可选）逻辑一致性项 I(D) 计算 U(D)；
- 基于模拟退火，在子集 + 标签空间中搜索高 U(D) 的集合。

Notes / 注意：
- We do NOT implement Algorithm 2 ConsistencyFix from the paper.
  这里不实现论文中的 Algorithm 2 ConsistencyFix。

- Instead, the ICM main loop supports two modes controlled by
  `enforce_unique_cid`:
    * If enforce_unique_cid=True, we impose a hard K-subset constraint:
      each consistency_id appears at most once in the current subset D,
      implemented via initialization + move rules. This is a stricter
      variant that deviates from the original paper by structurally
      forbidding multiple answers per question in D.
    * If enforce_unique_cid=False, we allow multiple answers for the same
      question (same consistency_id) to co-exist in D. This is closer to
      the original paper-style behavior: we rely on the objective
      U(D) = alpha * P(D) - I(D) to decide whether such duplication is
      desirable, with I(D) penalizing logically inconsistent groups.
  相应地，ICM 主循环通过 `enforce_unique_cid` 提供两种模式：
    * 当 enforce_unique_cid=True 时，会施加一个更“硬”的 K 子集约束：
      在当前子集 D 中，每个 consistency_id 至多出现一次。该约束通过
      初始化策略与 swap move 规则共同实现，相当于“每题最多选一个答案”，
      这一做法比原论文更严格。
    * 当 enforce_unique_cid=False 时，允许同一问题（同一 consistency_id）
      的多个回答同时存在于子集 D 中，这一设定更接近原论文的做法：
      由目标函数 U(D) = alpha * P(D) - I(D) 自动判断这些重复是否有利，
      其中 I(D) 会对同一 group 内的逻辑冲突进行惩罚。
"""

import asyncio
import json
import math
import random
import os
from datetime import datetime

from tqdm import tqdm

from icm.mutual_predictability import score_mutual_predictability_async
from icm.logical_consistency import score_logical_consistency_async

from utils.env import get_env_api_key
from utils.data import load_dataset_maybe

# ============================================================
# Demo dataset for U(D) / U(D) 测试用示例数据
# ============================================================

DEMO_DATA = [
    {
        "question": (
            'Actor in film and TV, with distinctive looks and three marriages '
            'and five children, decades of fame and also known for his online '
            'presence and political views. "Donald" what?'
        ),
        "choice": "Sutherland.",
        "label": 1,
        "consistency_id": 788,
    },
    {
        "question": (
            'Actor in film and TV, with distinctive looks and three marriages '
            'and five children, decades of fame and also known for his online '
            'presence and political views. "Donald" what?'
        ),
        "choice": "Trump.",
        "label": 0,
        "consistency_id": 788,
    },
]

DEMO_MODEL = "meta-llama/Meta-Llama-3.1-405B"


def _resolve_project_root() -> str:
    """
    Resolve project root as parent of current file's directory.
    将项目根目录视为当前文件所在目录的上一级目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # icm/
    project_root = os.path.dirname(current_dir)
    return project_root


# ============================================================
# U(D) computation
# U(D) 计算：支持不同 MP 方法与不同 I(D) 规则
# ============================================================


async def compute_u_async(
    dataset,
    model: str,
    api_key: str,
    mp_method: str = "official",
    alpha: float = 1.0,
    timeout: float = 60.0,
    top_logprobs: int = 20,
    max_concurrent: int = 8,
    use_consistency_term: bool = True,
    use_consistency_fix: bool = False,
    debug: bool = False,
    database: bool = True,
    consistency_mode: str = "at_most_one_true",
) -> dict:
    """
    Asynchronously compute U(D) for a dataset D.
    异步计算一组数据 D 的 U(D)。

    Target form / 数学形式（目标形式）：
        U(D) = alpha * P_theta(D) - I(D)

    Our implementation of P_theta(D):
        - We first get a scalar mutual-predictability score s_i for each
          example i via score_mutual_predictability_async.
        - If an example carries an ICM label y_i in {0, 1}, we treat it
          as a signed contribution:
                y_i = 1  ->  contrib_i = +s_i
                y_i = 0  ->  contrib_i = -s_i
          i.e., contrib_i = (2 * y_i - 1) * s_i.
        - If the example has no label (y_i is None or absent), we treat it
          as a neutral positive contribution contrib_i = +s_i (for backward
          compatibility with non-ICM use cases).
        - Then we sum all contrib_i to obtain P_theta(D).
          （可选：也可以除以样本数变成平均值，但这里保持求和形式，由 alpha
           控制整体缩放。）

    本实现中 P_theta(D) 的定义：
        - 先通过 score_mutual_predictability_async 为每条样本 i 获取
          一个标量的 mutual predictability 分数 s_i。
        - 若样本带有 ICM 内部使用的标签 y_i ∈ {0,1}，则将其看作有符号贡献：
                y_i = 1  ->  contrib_i = +s_i
                y_i = 0  ->  contrib_i = -s_i
          即 contrib_i = (2 * y_i - 1) * s_i。
        - 若该样本没有 label（y_i 为 None 或不存在），则视作“中性正向”
          贡献：contrib_i = +s_i（用于兼容非 ICM 场景）。
        - 将所有 contrib_i 相加得到 P_theta(D)。
          （可选：也可以除以样本数变为平均值，此处保持求和形式，由 alpha
           控制整体缩放。）

    Current implementation / 当前实现：
        - P_theta(D) is always approximated by label-signed Mutual
          Predictability scores as described above.
          P_theta(D) 始终按上述方式由带符号的 Mutual Predictability 分数
          近似。
        - If use_consistency_term=True, we call logical_consistency.py
          to compute I(D), then U(D) = alpha * P - I.
          当 use_consistency_term=True 时，会调用 logical_consistency.py
          计算 I(D)，然后 U(D) = alpha * P - I。
        - If use_consistency_term=False, U(D) = alpha * P (pure MP ablation).
          当 use_consistency_term=False 时，只计算 U(D) = alpha * P，
          即只考虑 Mutual Predictability 的 ablation 设置。

    The use_consistency_fix flag is reserved for future integration of
    Algorithm 2 "ConsistencyFix" in an outer loop; it is not used here.
    use_consistency_fix 参数目前仅占位，用于将来在更高层实现
    Algorithm 2 ConsistencyFix，本函数内部不使用。

    Args / 参数:
        dataset:
            Iterable of examples. Each example is a dict and must contain:
              - "question": str
              - "choice": str
            If use_consistency_term=True, each example is also expected to contain:
              - "label": int or bool
              - "consistency_id": hashable
            一组样本，可迭代对象。每个样本是一个 dict：
              - 必须包含 "question" 和 "choice"；
              - 如果 use_consistency_term=True，则还应包含
                "label" 与 "consistency_id" 字段。

        model:
            Hyperbolic model name, e.g. "meta-llama/Meta-Llama-3.1-405B".
            Hyperbolic 模型名称，例如 "meta-llama/Meta-Llama-3.1-405B"。

        api_key:
            Hyperbolic API key.
            Hyperbolic 的 API 密钥。

        mp_method:
            Mutual Predictability variant: "official" or "ll_stub".
            Mutual Predictability 的变体："official" 或 "ll_stub"。

        alpha:
            Weight for P_theta(D) in U(D) = alpha * P - I.
            U(D) 中 Mutual Predictability 部分的系数 alpha。

        timeout:
            Per-request timeout in seconds.
            每次模型调用的超时时间（秒）。

        top_logprobs:
            Number of top-k logprobs per token.
            模型返回的 top_k logprobs 数量。

        max_concurrent:
            Max number of concurrent model calls (via asyncio.Semaphore).
            最大并发模型调用数量（通过 asyncio.Semaphore 控制）。

        use_consistency_term:
            Whether to include I(D) in U(D).
            是否在 U(D) 中包含逻辑一致性项 I(D)。

        use_consistency_fix:
            Reserved flag for Algorithm 2 ConsistencyFix; not used here.
            为后续实现 Algorithm 2 ConsistencyFix 预留的开关，此处不使用。

        debug:
            Whether to enable debug logging in lower-level calls.
            是否在底层调用中开启 debug。

        database:
            Whether to enable sqlite caching in Hyperbolic helper.
            是否启用 Hyperbolic 封装中的 sqlite 缓存。

        consistency_mode:
            How to compute I(D) when use_consistency_term=True.
            当 use_consistency_term=True 时，如何计算 I(D)。

            - "at_most_one_true":
                Use AtMostOneTruePerConsistencyIdRule (penalize extra Trues
                in the same group).
                使用 AtMostOneTruePerConsistencyIdRule（同组中多出的 True 都记入惩罚）。

            - "conflict_count":
                Use TrueFalseConflictPerConsistencyIdRule (penalize each group
                where both True and False appear).
                使用 TrueFalseConflictPerConsistencyIdRule（同组内 True/False 同时出现则记 1 次冲突）。

    Returns / 返回:
        dict:
            {
                "U": float,                   # U(D)
                "P": float,                   # P_theta(D) = sum_i signed s_i
                "I": float,                   # logical consistency penalty I(D)
                "alpha": float,
                "mp_method": str,             # "official" or "ll_stub"
                "use_consistency_term": bool,
                "use_consistency_fix": bool,  # unused placeholder
                "consistency_mode": str,      # I(D) rule used
                "per_example": [
                    {
                        **original_example_fields,
                        "mp_score": float,    # per-example MP score s_i
                        "mp_raw": dict,       # raw MP response
                    },
                    ...
                ],
            }
    """
    if mp_method not in ("official", "ll_stub"):
        raise ValueError(
            'mp_method must be "official" or "ll_stub". / '
            'mp_method 参数必须是 "official" 或 "ll_stub"。'
        )

    # Convert to list for multiple passes and index-based access.
    # 转为 list，方便多次遍历和按下标访问。
    dataset = list(dataset)

    # Basic field check: question / choice must exist.
    # 基本字段校验：question / choice 必须存在。
    for i, ex in enumerate(dataset):
        if "question" not in ex or "choice" not in ex:
            raise KeyError(
                f"Example {i} missing 'question' or 'choice' field. / "
                f"第 {i} 条样本缺少 'question' 或 'choice' 字段。"
            )

    # If I(D) is enabled, we require label and consistency_id.
    # 如果需要逻辑一致性项，则要求 label 和 consistency_id。
    if use_consistency_term:
        for i, ex in enumerate(dataset):
            if "label" not in ex or "consistency_id" not in ex:
                raise KeyError(
                    f"Example {i} missing 'label' or 'consistency_id' field "
                    f"while use_consistency_term=True. / "
                    f"当 use_consistency_term=True 时，第 {i} 条样本缺少 "
                    f"'label' 或 'consistency_id' 字段。"
                )

    # Empty dataset: return zeros.
    # 空数据集直接返回 0。
    if not dataset:
        return {
            "U": 0.0,
            "P": 0.0,
            "I": 0.0,
            "alpha": alpha,
            "mp_method": mp_method,
            "use_consistency_term": use_consistency_term,
            "use_consistency_fix": use_consistency_fix,
            "consistency_mode": consistency_mode,
            "per_example": [],
        }

    # Use a semaphore to limit concurrency.
    # 使用信号量限制并发数。
    sem = asyncio.Semaphore(max_concurrent)

    async def _score_one(_index: int, _ex: dict):
        """
        Score a single example with Mutual Predictability.
        对单条样本进行 Mutual Predictability 打分。
        """
        question = _ex["question"]
        choice = _ex["choice"]

        async with sem:
            mp_result = await score_mutual_predictability_async(
                question=question,
                choice=choice,
                model=model,
                api_key=api_key,
                method=mp_method,
                timeout=timeout,
                top_logprobs=top_logprobs,
                debug=debug,
                database=database,
            )

        score = float(mp_result["score"])
        return _index, score, mp_result

    # Submit all tasks concurrently.
    # 并发提交所有任务。
    tasks = []
    for i, ex in enumerate(dataset):
        tasks.append(asyncio.create_task(_score_one(i, ex)))

    results = await asyncio.gather(*tasks)

    # Collect per-example results and accumulate P(D).
    # 按原顺序整理结果，并累加 P(D)。
    per_example = [None] * len(dataset)
    p_value = 0.0

    for index, s_i, mp_raw in results:
        ex_copy = dict(dataset[index])

        # Determine label-based sign for this example's contribution.
        # 根据当前样本的 label 决定其在 P(D) 中的符号权重。
        label_val = ex_copy.get("label", None)

        if label_val is None:
            # If no label is present (e.g. non-ICM use), treat as +1.
            # 若不存在 label（例如非 ICM 场景），视作 +1 贡献。
            weight = 1.0
        else:
            # Map label to sign: 1 -> +1, 0 -> -1.
            # 将标签映射为符号：1 -> +1，0 -> -1。
            weight = 1.0 if bool(label_val) else -1.0

        signed_score = weight * s_i
        p_value += signed_score

        ex_copy["mp_score"] = s_i
        ex_copy["mp_raw"] = mp_raw
        per_example[index] = ex_copy

    # Default I(D)=0 when logical consistency term is disabled.
    # 默认不使用逻辑一致性项时，I(D)=0。
    i_value = 0.0

    if use_consistency_term:
        # Build examples_for_logic and labels_for_logic:
        #   examples_for_logic: need only "id" + "consistency_id".
        #   labels_for_logic: id -> bool.
        # 构造 examples_for_logic 与 labels_for_logic：
        #   examples_for_logic：只需 "id" + "consistency_id"；
        #   labels_for_logic：id -> bool。
        examples_for_logic = []
        labels_for_logic = {}

        for idx, ex in enumerate(dataset):
            if "id" in ex:
                ex_id = ex["id"]
            else:
                ex_id = idx

            cid = ex["consistency_id"]

            examples_for_logic.append(
                {
                    "id": ex_id,
                    "consistency_id": cid,
                }
            )

            # Convert labels to bool: 0 -> False, non-zero -> True.
            # 将 label 转为 bool：0 -> False，其余视为 True。
            label_val = ex.get("label", 0)
            labels_for_logic[ex_id] = bool(label_val)

        # Compute logical consistency penalty I(D).
        # 计算逻辑一致性惩罚值 I(D)。
        i_value = await score_logical_consistency_async(
            examples_for_logic,
            labels_for_logic,
            mode=consistency_mode,
        )

    # Compute U(D).
    # 计算 U(D)。
    if use_consistency_term:
        u_value = alpha * p_value - i_value
    else:
        u_value = alpha * p_value

    return {
        "U": u_value,
        "P": p_value,
        "I": float(i_value),
        "alpha": alpha,
        "mp_method": mp_method,
        "use_consistency_term": use_consistency_term,
        "use_consistency_fix": use_consistency_fix,
        "consistency_mode": consistency_mode,
        "per_example": per_example,
    }


def compute_u(
    dataset,
    model: str,
    api_key: str,
    mp_method: str = "official",
    alpha: float = 1.0,
    timeout: float = 60.0,
    top_logprobs: int = 20,
    max_concurrent: int = 8,
    use_consistency_term: bool = True,
    use_consistency_fix: bool = False,
    debug: bool = False,
    database: bool = True,
    consistency_mode: str = "at_most_one_true",
) -> dict:
    """
    Synchronous wrapper around compute_u_async.
    compute_u_async 的同步包装，方便在普通脚本中直接调用 U(D)。
    """
    return asyncio.run(
        compute_u_async(
            dataset=dataset,
            model=model,
            api_key=api_key,
            mp_method=mp_method,
            alpha=alpha,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_concurrent=max_concurrent,
            use_consistency_term=use_consistency_term,
            use_consistency_fix=use_consistency_fix,
            debug=debug,
            database=database,
            consistency_mode=consistency_mode,
        )
    )


def get_temperature(
    iteration, initial_temp, final_temp, decay_rate, schedule="exp"
):
    """
    Calculate the temperature for simulated annealing.
    计算模拟退火过程中的当前温度。

    Parameters / 参数:
        iteration:
            Current iteration index (0, 1, 2, ...).
            当前迭代编号（0,1,2,...）。

        initial_temp:
            Initial temperature.
            初始温度。

        final_temp:
            Minimum temperature (lower bound).
            最终温度下限。

        decay_rate:
            Exponential decay rate (only when schedule="exp").
            指数衰减时的衰减系数（仅 schedule="exp" 时使用）。

        schedule:
            "exp" or "log".
            "exp" 或 "log"。

    Returns / 返回:
        float: current temperature (never below final_temp).
        float: 当前温度（不低于 final_temp）。
    """
    if schedule == "exp":
        # Exponential decay: T_k = max(T_min, T0 * decay_rate^k).
        # 指数衰减：T_k = max(T_min, T0 * decay_rate^k)。
        return max(final_temp, initial_temp * (decay_rate ** iteration))
    elif schedule == "log":
        # Logarithmic decay: T_k = max(T_min, T0 / (1 + 2 log(1 + k))).
        # 对数衰减：T_k = max(T_min, T0 / (1 + 2 log(1 + k)))。
        return max(final_temp, initial_temp / (1 + 2 * math.log(1 + iteration)))
    else:
        raise ValueError("schedule must be 'exp' or 'log'.")


# ============================================================
# ICM main loop: simulated annealing over subsets + labels
# ICM 主循环：基于模拟退火，在子集 + 标签空间中搜索
# ============================================================


def icm_main(
    data,
    model: str,
    api_key: str = None,
    mp_method: str = "official",
    alpha: float = 1.0,
    target_subset_size: int = 8,
    max_iter: int = 500,
    initial_t: float = 5.0,
    final_t: float = 0.1,
    decay: float = 0.98,
    scheduler: str = "log",
    use_consistency_term: bool = True,
    timeout: float = 60.0,
    top_logprobs: int = 20,
    max_concurrent: int = 8,
    save_result: bool = True,
    result_prefix: str = "icm_result",
    seed: int = None,
    debug: bool = False,
    consistency_mode: str = "at_most_one_true",
    enforce_unique_cid: bool = False,
    result_root: str | None = None,
):
    """
    ICM main loop (simulated annealing over subset + labels).
    ICM 主循环（基于模拟退火，在子集 + 标签空间上搜索）。

    State representation / 状态表示:
        - working_data: list of length N, the full dataset.
          working_data：长度为 N 的列表，对应完整数据集。
        - each example ex:
            - ex["id"]: global unique id.
              ex["id"]：全局唯一 id。
            - ex["label"] is None: not in current subset D.
              ex["label"] 为 None：当前不在子集 D 中。
            - ex["label"] in {0, 1}: in current subset D with an assigned label.
              ex["label"] 为 0/1：在当前子集 D 中，并带有标签。

    Hard vs soft constraint on consistency_id / consistency_id 的硬约束与软约束:

        - When enforce_unique_cid=True:
            We impose a hard constraint: at any time, in the current subset D,
            each consistency_id appears at most once. That is, for each question
            we select at most one answer. This is implemented by the
            initialization strategy and the swap-move rules, and acts as a
            stricter K-subset selection variant (not the original paper's
            default behavior).
          当 enforce_unique_cid=True 时：
            我们施加硬约束：在任意时刻，当前子集 D 中同一个 consistency_id
            至多出现一次，也就是说，每个问题在 D 中最多选中一个回答。
            该约束由初始化策略和 swap move 的规则共同保证，本质上是一个
            “每题最多一个回答”的 K 子集选择变体，并非原论文的默认设定，
            而是更严格的版本。

        - When enforce_unique_cid=False:
            We do NOT enforce uniqueness per consistency_id. Multiple answers
            to the same question (same consistency_id) are allowed to co-exist
            in D. This is closer to the original paper-style behavior: whether
            such duplication is desirable is decided automatically by the
            overall objective U(D) = alpha * P(D) - I(D), where I(D) penalizes
            logical conflicts within the same consistency group.
          当 enforce_unique_cid=False 时：
            我们不再强制每个 consistency_id 只出现一次。允许同一问题
           （相同的 consistency_id）的多个回答同时存在于子集 D 中。
            这一设定更接近原论文的做法：是否保留这些重复样本由总体目标函数
            U(D) = alpha * P(D) - I(D) 自动决定，其中 I(D) 会对同一组内的
            逻辑冲突进行惩罚。

    Move types / Move 类型:

        1. Flip move:
            - Pick a random example i in D.
            - Flip label: 0 -> 1, or 1 -> 0.
            - Subset membership is unchanged.
            - 从 D 中随机选一个样本 i，将其标签从 0 翻转为 1，或从 1 翻转为 0。
              不改变该样本是否属于子集 D，仅改变其标签。

        2. Swap move:
            - When enforce_unique_cid=True:
                * Pick a random example i_in in D.
                * Pick j_out from the full dataset with label=None such that:
                    - cid(j_out) == cid(i_in), or
                    - cid(j_out) is a new group not currently in D.
                * Set i_in.label = None (remove from D),
                  set j_out.label = random{0,1} (add to D).
              当 enforce_unique_cid=True 时：
                * 从 D 中随机选一个样本 i_in。
                * 再从全集中选一个当前 label=None 的样本 j_out，使得：
                    - j_out 的 consistency_id 等于 i_in 的 consistency_id（组内换人），或
                    - j_out 的 consistency_id 在当前 D 中不存在（引入新组）。
                * 令 i_in.label = None（从 D 中移除），
                  再将 j_out.label 随机设为 0 或 1（加入 D）。

            - When enforce_unique_cid=False:
                * Pick a random example i_in in D.
                * Pick any j_out from the full dataset with label=None
                  (no restriction on consistency_id).
                * Set i_in.label = None (remove from D),
                  set j_out.label = random{0,1} (add to D).
              当 enforce_unique_cid=False 时：
                * 从 D 中随机选一个样本 i_in。
                * 从全集中任意选一个当前 label=None 的样本 j_out，
                  不再对 consistency_id 施加任何限制。
                * 令 i_in.label = None（从 D 中移除），
                  再将 j_out.label 随机设为 0 或 1（加入 D）。

    Objective / 目标函数:
        - For the current subset D, call compute_u(...) to get U(D),
          where P(D) is a label-signed sum of per-example MP scores and
          I(D) is an optional logical consistency penalty.
        - For a candidate subset D_hat, also call compute_u(...) to get U(D_hat).
        - Accept or reject the move via the Metropolis rule, based on
          ΔU = U(D_hat) - U(D) and the current temperature T.
        - 对当前子集 D 调用 compute_u(...) 得到 U(D)，其中 P(D) 是按标签
          带符号的 MP 得分之和，I(D) 为可选的逻辑一致性惩罚项。
        - 对候选子集 D_hat 同样调用 compute_u(...) 得到 U(D_hat)，
          然后根据 ΔU = U(D_hat) - U(D) 与当前温度 T，
          使用 Metropolis 规则决定是否接受该 move。

    Args / 参数:
        data:
            list[dict] or JSON file path.
            必须至少包含字段：
                - "question"
                - "choice"
                - "consistency_id"（若 use_consistency_term=True）。
            原始 TruthfulQA 中的 "label" 会被复制到 "gold_label"，
            ICM 内部使用的标签写在 "label" 字段中。

        model:
            Hyperbolic model name, e.g. DEMO_MODEL.
            Hyperbolic 模型名称，例如 DEMO_MODEL。

        api_key:
            Hyperbolic API Key; if None, read from env via get_env_api_key().
            Hyperbolic API Key；若为 None，则调用 get_env_api_key() 从环境变量读取。

        mp_method:
            Mutual Predictability variant: "official" or "ll_stub".
            Mutual Predictability 使用的方法："official" 或 "ll_stub"。

        alpha:
            Weight alpha in U(D) = alpha * P(D) - I(D).
            U(D) = alpha * P(D) - I(D) 中 Mutual Predictability 部分的权重 alpha。

        target_subset_size:
            Fixed subset size K, default 8.
            ICM 子集 D 的大小 K（固定不变），默认值为 8。

        max_iter:
            Max iterations for simulated annealing.
            模拟退火的最大迭代次数。

        initial_t, final_t, decay, scheduler:
            Temperature schedule parameters, passed to get_temperature().
            温度调度相关参数，通过 get_temperature() 计算每一步的温度 T。

        use_consistency_term:
            Whether to include logical consistency penalty I(D).
            是否在 U(D) 中加入逻辑一致性惩罚项 I(D)。

        timeout, top_logprobs, max_concurrent:
            Passed down to compute_u / score_mutual_predictability_async.
            将这些参数透传给 compute_u / score_mutual_predictability_async。

        save_result:
            Whether to dump final subset D and U(D) trace to JSON files.
            是否将最终子集 D 以及 U(D) 轨迹保存为 JSON 文件。

        result_prefix:
            Prefix for result file name or folder.
            结果文件（夹）名的前缀，例如 "icm_truthfulqa_K8"。

        seed:
            Random seed for reproducibility.
            随机种子，用于复现实验结果。

        debug:
            Whether to enable debug in compute_u (and below).
            是否在 compute_u 及其下游调用中开启 debug 输出。

        consistency_mode:
            Which I(D) rule to use when use_consistency_term=True.
            当 use_consistency_term=True 时使用哪种 I(D) 规则，
            例如 "at_most_one_true" 或 "conflict_count"。

        enforce_unique_cid:
            If True, enforce a hard constraint that each consistency_id
            appears at most once in D via the initialization and swap rules.
            If False, do not enforce uniqueness and rely on I(D) to penalize
            logically inconsistent groups (soft constraint, closer to paper).
            若为 True，则通过初始化与 swap 规则实现硬约束：
            每个 consistency_id 在 D 中至多出现一次。
            若为 False，则不再强制唯一性，由 I(D) 对逻辑不一致的 group
            进行惩罚（软约束，更接近原论文的设定）。

        result_root:
            If provided, save target_subset.json, ud.json and icm_arguments.json
            directly under this directory. If None, fall back to the legacy
            layout {project_root}/results/RP_{...}/.
            若提供，则直接在该目录下保存 target_subset.json、ud.json
            与 icm_arguments.json；若为 None，则回退到旧的
            {项目根目录}/results/RP_{...}/ 布局。
    """
    run_start_time = datetime.now().isoformat()  # 记录 ICM 运行开始时间

    # Set random seed.
    # 设置随机种子。
    if seed is not None:
        random.seed(seed)

    # Load data.
    # 加载数据。
    raw_data = load_dataset_maybe(data)
    if not raw_data:
        raise ValueError("Empty dataset passed to icm_main. / 传入 icm_main 的数据集为空。")

    if isinstance(data, str):
        print(f"[ICM] loaded dataset from '{data}', size = {len(raw_data)}")
    else:
        print(f"[ICM] loaded dataset from in-memory list, size = {len(raw_data)}")

    # Get API key from env if not provided.
    # 若未提供 api_key，则从环境变量读取。
    if api_key is None:
        api_key = get_env_api_key()

    # Make a working copy so we don't mutate the original data.
    # 复制一份工作数据，确保不修改原始对象。
    working_data = []
    for idx, ex in enumerate(raw_data):
        ex_copy = dict(ex)

        # Normalize id field: use existing id if present, else index.
        # 统一 id 字段：若已有 id 就用原来的，否则用索引。
        if "id" not in ex_copy:
            ex_copy["id"] = idx

        # Preserve original dataset label as gold_label if present.
        # 保存原始 TruthfulQA 标签（若有）到 gold_label。
        if "label" in ex_copy:
            ex_copy["gold_label"] = ex_copy["label"]

        # ICM label used in the search, initially None.
        # ICM 搜索中使用的标签写在 label 字段中，初始为 None。
        ex_copy["label"] = None

        working_data.append(ex_copy)

    n = len(working_data)
    if target_subset_size > n:
        target_subset_size = n

    print(
        f"[ICM] N = {n}, "
        f"target_subset_size = {target_subset_size}, "
        f"max_iter = {max_iter}, "
        f"enforce_unique_cid = {enforce_unique_cid}"
    )

    # ============================================================
    # Initialization of subset D / 子集 D 的初始化
    # ============================================================
    if enforce_unique_cid:
        # Hard-constraint initialization:
        #   - Group examples by consistency_id.
        #   - Randomly select target_subset_size different consistency_ids.
        #   - For each group, pick one example, assign random label 0/1.
        #
        # 硬约束初始化：
        #   - 按 consistency_id 分组；
        #   - 随机选出 target_subset_size 个不同的 consistency_id；
        #   - 每组随机选 1 条样本，随机赋标签 0/1。
        groups_by_cid = {}
        for idx, ex in enumerate(working_data):
            cid = ex.get("consistency_id", idx)
            groups_by_cid.setdefault(cid, []).append(idx)

        all_cids = list(groups_by_cid.keys())
        if target_subset_size > len(all_cids):
            target_subset_size = len(all_cids)

        chosen_cids = random.sample(all_cids, target_subset_size)
        for cid in chosen_cids:
            idx = random.choice(groups_by_cid[cid])
            working_data[idx]["label"] = random.randint(0, 1)
    else:
        # Soft-constraint initialization:
        #   - Simply sample K indices without grouping by consistency_id.
        #   - Multiple answers to the same question can be included in D.
        #
        # 软约束初始化：
        #   - 不按 consistency_id 分组，直接随机采样 K 条样本；
        #   - 同一问题的多条回答可以同时进入 D。
        all_indices = list(range(n))
        if target_subset_size > 0:
            init_indices = random.sample(all_indices, target_subset_size)
            for idx in init_indices:
                working_data[idx]["label"] = random.randint(0, 1)

    # Helper: build current subset D from working_data (label!=None),
    # and record all ids that have ever appeared in D (for coverage stats).
    # 辅助函数：从 working_data 中收集当前子集 D（label!=None），
    # 并记录曾经出现在 D 中的所有 id。
    seen_ids = set()

    def build_current_subset():
        subset = [_ex for _ex in working_data if _ex.get("label") is not None]
        for _ex in subset:
            seen_ids.add(_ex["id"])
        return subset

    # --------------------------------------------------------
    # Compute initial U(D) and initialize U(D) trajectory.
    # 计算初始 U(D)，并初始化 U(D) 轨迹记录。
    # --------------------------------------------------------
    current_subset = build_current_subset()
    current_result = compute_u(
        dataset=current_subset,
        model=model,
        api_key=api_key,
        mp_method=mp_method,
        alpha=alpha,
        timeout=timeout,
        top_logprobs=top_logprobs,
        max_concurrent=max_concurrent,
        use_consistency_term=use_consistency_term,
        use_consistency_fix=False,
        debug=debug,
        database=True,
        consistency_mode=consistency_mode,
    )
    current_u = current_result["U"]
    best_u = current_u
    best_snapshot = [dict(ex) for ex in working_data]

    # U(D) history for plotting later.
    # 记录 U(D) 的变化轨迹，方便后续绘图。
    u_trace = []
    u_trace.append(
        {
            "iteration": 0,
            "temperature": None,
            "prev_U": None,
            "candidate_U": current_u,
            "new_U": current_u,
            "candidate_P": current_result.get("P", None),
            "candidate_I": current_result.get("I", None),
            "move_type": "init",
            "accepted": True,
        }
    )

    # Stats.
    # 统计信息。
    total_u_eval = 1  # We already computed U(D) once.
    accepted_moves = 0
    rejected_moves = 0
    flip_moves_tried = 0
    flip_moves_accepted = 0
    swap_moves_tried = 0
    swap_moves_accepted = 0

    # Helper: indices of examples currently in D (label!=None).
    # 辅助函数：当前子集 D 中样本的下标列表（label!=None）。
    def labeled_indices():
        return [_i for _i, _ex in enumerate(working_data) if _ex.get("label") is not None]

    # Main simulated annealing loop.
    # 模拟退火主循环。
    for it in tqdm(range(max_iter), desc="ICM (simulated annealing)"):
        # Current temperature.
        # 当前温度。
        t = get_temperature(
            iteration=it,
            initial_temp=initial_t,
            final_temp=final_t,
            decay_rate=decay,
            schedule=scheduler,
        )

        labeled_idx_list = labeled_indices()
        if not labeled_idx_list:
            # Theoretically should not happen since we try to maintain subset size.
            # 理论上不应发生（我们始终尝试保持子集大小），防御性退出。
            break

        # Randomly choose move type: flip or swap (50% / 50%).
        # 随机选择 move 类型：flip 与 swap 各占 50% 概率。
        move_type = "flip" if random.random() < 0.5 else "swap"

        # Variables used to restore state if move rejected.
        # 若拒绝 move，用于恢复状态的变量。
        ex = None
        ex_in = None
        ex_out = None
        old_label = None
        old_label_in = None
        old_label_out = None

        if move_type == "flip":
            flip_moves_tried += 1

            # Flip: pick a random example in D and flip its label.
            # flip：从 D 中随机选一条样本翻转其标签。
            i = random.choice(labeled_idx_list)
            ex = working_data[i]
            old_label = ex["label"]
            new_label = 1 - old_label  # 0 <-> 1

            ex["label"] = new_label
            candidate_subset = build_current_subset()

        else:
            swap_moves_tried += 1

            if enforce_unique_cid:
                # Hard-constraint swap:
                #   - Remove one example i_in from D.
                #   - Add j_out from outside D such that
                #       * cid_out == cid_in (same group), or
                #       * cid_out not in cids_in_d (new group).
                #
                # 硬约束 swap：
                #   - 从 D 中移除一个 i_in；
                #   - 从 D 外选择 j_out，满足：
                #       * cid_out == cid_in（同组内换人），或
                #       * cid_out 不在当前 D 的 group 中（引入新组）。
                i_in = random.choice(labeled_idx_list)
                ex_in = working_data[i_in]
                cid_in = ex_in.get("consistency_id", i_in)

                cids_in_d = set(
                    working_data[k].get("consistency_id", k)
                    for k in labeled_idx_list
                )

                candidate_out_indices = []
                for j, ex_out_candidate in enumerate(working_data):
                    if ex_out_candidate.get("label") is not None:
                        continue
                    cid_out = ex_out_candidate.get("consistency_id", j)
                    if (cid_out == cid_in) or (cid_out not in cids_in_d):
                        candidate_out_indices.append(j)

                if not candidate_out_indices:
                    # No valid swap move this iteration; skip.
                    # 本轮不存在合法 swap，则跳过，不重新计算 U(D)。
                    continue

                j_out = random.choice(candidate_out_indices)
                ex_out = working_data[j_out]

                old_label_in = ex_in["label"]         # 0 or 1
                old_label_out = ex_out.get("label")   # None

                ex_in["label"] = None
                ex_out["label"] = random.randint(0, 1)

                candidate_subset = build_current_subset()

            else:
                # Soft-constraint swap:
                #   - Remove one example i_in from D.
                #   - Add any j_out from outside D (label=None),
                #     regardless of consistency_id.
                #
                # 软约束 swap：
                #   - 从 D 中移除一个 i_in；
                #   - 从 D 外任意选择一个 label=None 的 j_out，
                #     不再限制 consistency_id。
                i_in = random.choice(labeled_idx_list)
                ex_in = working_data[i_in]

                candidate_out_indices = [
                    j
                    for j, ex_out_candidate in enumerate(working_data)
                    if ex_out_candidate.get("label") is None
                ]
                if not candidate_out_indices:
                    # No unused sample; cannot swap.
                    # 没有未选中的样本，无法 swap，本轮跳过。
                    continue

                j_out = random.choice(candidate_out_indices)
                ex_out = working_data[j_out]

                old_label_in = ex_in["label"]         # 0 or 1
                old_label_out = ex_out.get("label")   # None

                ex_in["label"] = None
                ex_out["label"] = random.randint(0, 1)

                candidate_subset = build_current_subset()

        # Compute U(D_hat).
        # 计算候选子集的 U(D_hat)。
        candidate_result = compute_u(
            dataset=candidate_subset,
            model=model,
            api_key=api_key,
            mp_method=mp_method,
            alpha=alpha,
            timeout=timeout,
            top_logprobs=top_logprobs,
            max_concurrent=max_concurrent,
            use_consistency_term=use_consistency_term,
            use_consistency_fix=False,
            debug=debug,
            database=True,
            consistency_mode=consistency_mode,
        )
        candidate_u = candidate_result["U"]
        total_u_eval += 1

        # ΔU = U(D_hat) - U(D).
        # 计算 ΔU = U(D_hat) - U(D)。
        prev_u = current_u
        delta_u = candidate_u - current_u

        # Metropolis acceptance.
        # Metropolis 接受判定。
        accept = False
        if delta_u >= 0:
            accept = True
        else:
            # Accept with probability exp(ΔU / T).
            # 以概率 exp(ΔU / T) 接受。
            prob = math.exp(delta_u / max(t, 1e-8))
            if random.random() < prob:
                accept = True

        if accept:
            # Accept new state.
            # 接受新状态。
            current_u = candidate_u
            accepted_moves += 1

            if move_type == "flip":
                flip_moves_accepted += 1
            else:
                swap_moves_accepted += 1

            # Update the best snapshot if improved.
            # 若当前更优，则更新最佳快照。
            if current_u > best_u:
                best_u = current_u
                best_snapshot = [dict(ex) for ex in working_data]
        else:
            # Reject and restore previous state.
            # 拒绝候选状态，恢复上一状态。
            rejected_moves += 1
            if move_type == "flip":
                ex["label"] = old_label
            else:
                ex_in["label"] = old_label_in
                ex_out["label"] = old_label_out

        # ----------------------------------------------------
        # Record U(D) trajectory for this iteration.
        # 记录本次迭代的 U(D) 变化（prev / candidate / new）。
        # ----------------------------------------------------
        u_trace.append(
            {
                "iteration": it + 1,  # 从 1 开始计迭代步，0 是初始化
                "temperature": t,
                "prev_U": prev_u,
                "candidate_U": candidate_u,
                "new_U": current_u,
                "candidate_P": candidate_result.get("P", None),
                "candidate_I": candidate_result.get("I", None),
                "move_type": move_type,
                "accepted": accept,
            }
        )

    # Use best snapshot as final result.
    # 最终采用 best_snapshot 中的状态。
    working_data = best_snapshot
    final_subset = [ex for ex in working_data if ex.get("label") is not None]

    # Print summary.
    # 打印汇总信息。
    print("========== ICM summary ==========")
    print(f"[ICM] total U(D) evaluations = {total_u_eval}")
    print(f"[ICM] total accepted moves    = {accepted_moves}")
    print(f"[ICM] total rejected moves    = {rejected_moves}")
    print(f"[ICM] flip moves: tried={flip_moves_tried}, accepted={flip_moves_accepted}")
    print(f"[ICM] swap moves: tried={swap_moves_tried}, accepted={swap_moves_accepted}")
    print(f"[ICM] total unique ids in D   = {len(seen_ids)} / N = {n}")

    # Save result if requested.
    # 如需保存结果到 JSON。
    if save_result:
        # ------------------------------------------------------------
        # Resolve folder_path based on result_root or legacy layout.
        # 基于 result_root 或旧布局解析结果目录路径。
        # ------------------------------------------------------------
        if result_root is not None:
            folder_path = result_root
            os.makedirs(folder_path, exist_ok=True)
        else:
            # Legacy layout under project_root/results/RP_...
            # 旧布局：保存在 project_root/results/RP_... 目录下。
            project_root = _resolve_project_root()
            results_dir = os.path.join(project_root, "results")
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = (
                f"RP_{result_prefix}_TS_{timestamp}_MM_{mp_method}_"
                f"CM_{consistency_mode}_UC_{enforce_unique_cid}"
            )
            folder_path = os.path.join(results_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

        # ------------------------------------------------------------
        # Save final subset as target_subset.json
        # 保存最终子集为 target_subset.json
        # ------------------------------------------------------------
        subset_path = os.path.join(folder_path, "target_subset.json")
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(final_subset, f, ensure_ascii=False, indent=2)
        print(f"[ICM] Target subset saved to: {subset_path}")

        # ------------------------------------------------------------
        # Save U(D) trajectory as ud.json
        # 保存 U(D) 变化轨迹为 ud.json
        # ------------------------------------------------------------
        ud_path = os.path.join(folder_path, "ud.json")
        with open(ud_path, "w", encoding="utf-8") as f:
            json.dump(u_trace, f, ensure_ascii=False, indent=2)
        print(f"[ICM] U(D) trace saved to: {ud_path}")

        # ------------------------------------------------------------
        # Save ICM arguments snapshot as icm_arguments.json
        # 保存 ICM 启动参数快照为 icm_arguments.json
        # ------------------------------------------------------------
        args_path = os.path.join(folder_path, "icm_arguments.json")
        icm_args = {
            "data": data if isinstance(data, str) else "<in-memory>",
            "model": model,
            "mp_method": mp_method,
            "alpha": alpha,
            "target_subset_size": target_subset_size,
            "max_iter": max_iter,
            "initial_t": initial_t,
            "final_t": final_t,
            "decay": decay,
            "scheduler": scheduler,
            "use_consistency_term": use_consistency_term,
            "timeout": timeout,
            "top_logprobs": top_logprobs,
            "max_concurrent": max_concurrent,
            "save_result": save_result,
            "result_prefix": result_prefix,
            "result_root": result_root,
            "seed": seed,
            "debug": debug,
            "consistency_mode": consistency_mode,
            "enforce_unique_cid": enforce_unique_cid,
            "run_start_time": run_start_time,
            "saved_time": datetime.now().isoformat(),
            "api_key_source": "env" if api_key is None else "provided_or_env",
        }
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(icm_args, f, ensure_ascii=False, indent=2)
        print(f"[ICM] Arguments saved to: {args_path}")

    return final_subset


# ============================================================
# Demos: U(D) and ICM on small data
# U(D) Demo 与小型 ICM Demo
# ============================================================


def run_u_demo():
    """
    Demo for computing U(D) with different MP and I(D) modes.
    使用不同 MP 方法与 I(D) 模式演示 U(D) 的计算结果。
    """
    model = DEMO_MODEL
    api_key = get_env_api_key()  # Read from env / 从环境变量读取

    print("=== U(D) with official MP, at_most_one_true I(D) ===")
    out1 = compute_u(
        dataset=DEMO_DATA,
        model=model,
        api_key=api_key,
        mp_method="official",
        alpha=1.0,
        max_concurrent=4,
        use_consistency_term=True,
        use_consistency_fix=False,
        debug=True,
        database=True,
        consistency_mode="at_most_one_true",
    )
    print("U(D) =", out1["U"])
    print("P(D) =", out1["P"])
    print("I(D) =", out1["I"])
    print("First example mp_score =", out1["per_example"][0]["mp_score"])

    print("\n=== U(D) with official MP, conflict_count I(D) ===")
    out2 = compute_u(
        dataset=DEMO_DATA,
        model=model,
        api_key=api_key,
        mp_method="official",
        alpha=1.0,
        max_concurrent=4,
        use_consistency_term=True,
        use_consistency_fix=False,
        debug=False,
        database=True,
        consistency_mode="conflict_count",
    )
    print("U(D) =", out2["U"])
    print("P(D) =", out2["P"])
    print("I(D) =", out2["I"])
    print("First example mp_score =", out2["per_example"][0]["mp_score"])


def run_icm_demo():
    """
    Run a small ICM simulated-annealing loop on TruthfulQA-style data.
    在 TruthfulQA 样式数据上运行一个小型 ICM 模拟退火 Demo。

    - target_subset_size = 8 (as in the paper).
      子集大小 K=8，与论文设置一致。
    - use compute_u as the objective (label-signed MP + optional I(D)).
      使用 compute_u 作为目标函数（按标签带符号的 MP + 可选 I(D)）。
    """
    project_root = _resolve_project_root()
    sample_data = os.path.join(project_root, "truthfulqa_train.json")
    model = DEMO_MODEL
    api_key = get_env_api_key()

    # Create an attempt folder with an 'icm' subfolder.
    # 创建一个带 icm 子目录的 attempt 目录。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt_root = os.path.join(project_root, "results", f"attempt_{timestamp}")
    icm_root = os.path.join(attempt_root, "icm")
    os.makedirs(icm_root, exist_ok=True)

    print(
        "Running ICM demo on TruthfulQA-like data... / 在 TruthfulQA 数据上运行 ICM Demo..."
    )
    print(f"[ICM-DEMO] Attempt folder: {attempt_root}")

    result_subset = icm_main(
        data=sample_data,
        model=model,
        api_key=api_key,
        mp_method="ll_stub",        # demo: use ll_stub or official MP
        alpha=1.0,
        target_subset_size=8,
        max_iter=256 * 25,
        initial_t=5.0,
        final_t=0.1,
        decay=0.98,
        scheduler="log",
        use_consistency_term=True,
        timeout=120.0,
        top_logprobs=20,
        max_concurrent=4,
        save_result=True,
        result_prefix="icm_demo",
        seed=42,
        debug=False,
        consistency_mode="at_most_one_true",
        enforce_unique_cid=False,   # demo: paper-style (no uniqueness constraint) / Demo：更贴近论文的无唯一性约束设置
        result_root=icm_root,
    )

    print("\nFinal labeled subset / 最终带标签子集：")
    for ex in result_subset:
        print(
            f"[label={ex['label']}, gold_label={ex.get('gold_label')}] "
            f"Q: {ex['question']} | choice: {ex['choice']}"
        )


if __name__ == "__main__":
    # You can toggle which demo to run.
    # 可以自行选择运行哪个 Demo。
    # run_u_demo()
    run_icm_demo()
