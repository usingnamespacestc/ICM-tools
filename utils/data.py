# -*- coding: utf-8 -*-
"""
Data loading helpers for the ICM project.
ICM 项目的数据加载辅助函数。
"""

import json
import os


# ============================================================
# Path resolving helper
# 路径解析辅助函数
# ============================================================


def resolve_data_path(path, extra_search_dirs=None):
    """
    Resolve a (possibly relative) data file path in a robust way.
    以更健壮的方式解析（可能是相对的）数据文件路径。

    Strategy / 策略:
      1. If path is absolute and exists, use it directly.
         若 path 为绝对路径且存在，直接返回。
      2. If path is relative, try the following locations in order:
         若为相对路径，按如下顺序依次尝试：
            - path as-is (relative to current working directory)
              原样 path（相对当前工作目录）
            - path relative to this file's directory (data.py 所在目录)
              相对于 data.py 所在目录
            - path relative to project root (parent of this file's directory)
              相对于项目根目录（data.py 的上一级目录）
            - path under project_root/icm
              项目根目录下的 icm 子目录
            - path under project_root/eval
              项目根目录下的 eval 子目录
            - path under project_root/data
              项目根目录下的 data 子目录
            - path under any directory in extra_search_dirs (if provided)
              以及 extra_search_dirs（如提供）中指定的目录

    Args / 参数:
        path:
            File path string, absolute or relative.
            文件路径字符串，可以是绝对或相对。
        extra_search_dirs:
            Optional list of extra directories to search.
            可选的额外搜索目录列表。

    Returns / 返回:
        Resolved absolute path to an existing file.
        指向存在文件的绝对路径。

    Raises / 异常:
        FileNotFoundError:
            If no candidate location contains the file.
            若所有候选路径都不存在该文件则抛出异常。
    """
    if not isinstance(path, str):
        raise TypeError(
            "resolve_data_path expects a string path. / "
            "resolve_data_path 期望接收字符串类型的路径。"
        )

    # Absolute path: just check existence.
    # 绝对路径：直接检测是否存在。
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(
            "Absolute path does not exist: {} / 绝对路径文件不存在：{}".format(path, path)
        )

    # Collect candidate paths.
    # 收集一系列候选路径。
    candidates = []

    # 1) As-is (relative to current working directory).
    # 1) 原样 path（相对当前工作目录）。
    candidates.append(path)

    # 2) Relative to this file (data.py) directory.
    # 2) 相对于本文件（data.py）所在目录。
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, path))

    # 3) Relative to project root (parent of data.py directory).
    # 3) 相对于项目根目录（data.py 的上一级目录）。
    project_root = os.path.dirname(here)
    candidates.append(os.path.join(project_root, path))

    # 4) Under project_root/icm, project_root/eval, project_root/data.
    # 4) 项目根目录下的 icm、eval、data 子目录。
    candidates.append(os.path.join(project_root, "icm", path))
    candidates.append(os.path.join(project_root, "eval", path))
    candidates.append(os.path.join(project_root, "data", path))

    # 5) Extra search dirs if provided.
    # 5) 额外搜索目录（如提供）。
    if extra_search_dirs:
        for d in extra_search_dirs:
            if d:
                candidates.append(os.path.join(d, path))

    # Return the first existing candidate.
    # 返回第一个存在的候选路径。
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # If none matched, raise an informative error.
    # 若所有候选路径都不存在，抛出带中英提示的异常。
    msg_lines = [
        "Data file not found: {}".format(path),
        "Searched the following locations: / 已尝试以下路径：",
    ]
    for cand in candidates:
        msg_lines.append("  - {}".format(os.path.abspath(cand)))

    raise FileNotFoundError("\n".join(msg_lines))


# ============================================================
# Dataset loading helper
# 数据集加载辅助函数
# ============================================================


def load_dataset_maybe(data_or_path, extra_search_dirs=None):
    """
    Load a dataset that can be either a Python list[dict] or a JSON file path.
    加载数据集，支持传入 Python list[dict] 或 JSON 文件路径。

    Behavior / 行为:
      - If data_or_path is a string:
            Treat it as a JSON file path, resolve it via resolve_data_path,
            and json.load it.
        若 data_or_path 为字符串：
            视为 JSON 文件路径，通过 resolve_data_path 解析后使用 json.load 读取。
      - Otherwise:
            Convert it to list and return (defensive copy).
        否则：
            将其转换为 list 并返回（防御性复制）。

    Args / 参数:
        data_or_path:
            list-like object or JSON file path.
            list 风格对象或 JSON 文件路径。
        extra_search_dirs:
            Optional extra directories to search for the file.
            可选的额外搜索目录列表。

    Returns / 返回:
        A list of examples (list of dicts).
        样本列表（字典列表）。
    """
    # If it's a string, treat as file path.
    # 若为字符串，则视为文件路径。
    if isinstance(data_or_path, str):
        resolved = resolve_data_path(data_or_path, extra_search_dirs=extra_search_dirs)
        with open(resolved, "r", encoding="utf-8") as f:
            return json.load(f)

    # Otherwise assume it's already an iterable of examples.
    # 否则假定其已经是一个可迭代的样本集合。
    return list(data_or_path)


