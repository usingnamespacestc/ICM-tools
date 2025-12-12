# -*- coding: utf-8 -*-
"""
Plot helpers for ICM experiments.
ICM 实验的绘图辅助脚本。

Current features / 当前功能:
- Visualize U(D) search logs (ud.json) for each ICM method under one attempt.
  对单次 attempt 中各个 ICM 方法的 ud.json 进行可视化。

Directory layout assumption / 目录结构假设
--------------------------------------
project_root/
    eval/
        plot_icm_ud.py
    icm/
        ...
    results/
        attempt_YYYYMMDD_HHMMSS/
            evaluation/
                ...
            icm/
                ll_stub/
                    ud.json
                    ...
                official/
                    ud.json
                    ...
                utfs/
                    ud.json
                    ...
            output.txt
            ...

Usage examples / 使用示例
------------------------
1) Standalone from project root / 在项目根目录直接运行:

    python -m eval.plot_icm_ud
    python -m eval.plot_icm_ud --attempt attempt_20251210_033308 --method ll_stub
    python -m eval.plot_icm_ud --save
    python -m eval.plot_icm_ud --save --output-dir plots

2) From Python / 在 Python 中调用:

    from eval.plot_icm_ud import plot_ud_for_method, plot_ud_for_all_methods

    # Plot one method of the latest attempt and show interactively.
    plot_ud_for_method(method="ll_stub")

    # Plot and save to file without showing.
    plot_ud_for_method(
        method="ll_stub",
        attempt_name="attempt_20251210_033308",
        save_path="plots/ll_stub_ud.png",
        show=False,
    )

    # Plot all methods of latest attempt, and save next to each ud.json
    plot_ud_for_all_methods(show=False)

    # Plot all methods and save into a central folder:
    plot_ud_for_all_methods(save_dir="plots", show=False)
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


# ============================================================
# Path helpers / 路径辅助函数
# ============================================================


def get_project_root() -> Path:
    """
    Resolve project root based on this file's location.

    Assumes layout:
        project_root/
            eval/
                plot_icm_ud.py
            icm/
            results/

    通过当前文件所在路径自动解析项目根目录，避免依赖工作目录。
    """
    this_dir = Path(__file__).resolve().parent  # eval/
    return this_dir.parent  # project_root/


def get_icm_results_root(project_root: Optional[Path] = None) -> Path:
    """
    Return the root directory where ICM results are stored.

    默认假设 ICM 结果保存在:
        <project_root>/results
    """
    if project_root is None:
        project_root = get_project_root()
    return project_root / "results"


def list_attempt_dirs(results_root: Optional[Path] = None) -> List[Path]:
    """
    List all attempt_* directories under results_root (non-recursive).

    返回按名称排序的 attempt 目录列表。
    """
    if results_root is None:
        results_root = get_icm_results_root()

    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found: {results_root} / 未找到 results 目录"
        )

    attempt_dirs: List[Path] = []
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith("attempt_"):
            attempt_dirs.append(child)

    attempt_dirs.sort(key=lambda p: p.name)
    return attempt_dirs


def get_latest_attempt_dir(results_root: Optional[Path] = None) -> Path:
    """
    Return the latest attempt_* directory under results_root by name.

    按目录名排序，返回最后一个（通常是最新的一次 attempt）。
    """
    attempts = list_attempt_dirs(results_root=results_root)
    if not attempts:
        raise RuntimeError(
            f"No attempt_* folders found under {results_root} / 未找到任何 attempt_* 目录"
        )
    return attempts[-1]


def get_ud_json_path(
    method: str,
    attempt_name: Optional[str] = None,
    results_root: Optional[Path] = None,
) -> Path:
    """
    Build the path to ud.json for a given method and attempt.

    期望目录结构:
        <results_root>/
            attempt_xxx/
                icm/
                    <method>/
                        ud.json

    method 示例: "ll_stub", "official", "utfs" 等。
    """
    if results_root is None:
        results_root = get_icm_results_root()

    if attempt_name is None:
        attempt_dir = get_latest_attempt_dir(results_root=results_root)
    else:
        attempt_dir = results_root / attempt_name
        if not attempt_dir.exists():
            raise FileNotFoundError(
                f"Attempt folder not found: {attempt_dir} / 未找到指定 attempt 目录"
            )

    ud_path = attempt_dir / "icm" / method / "ud.json"
    if not ud_path.exists():
        raise FileNotFoundError(
            f"ud.json not found for method '{method}' under {attempt_dir}.\n"
            f"Expected path / 期望路径: {ud_path}"
        )
    return ud_path


# ============================================================
# Core plotting logic / 核心绘图逻辑
# ============================================================


def load_ud_records(ud_json_path: Path) -> List[dict]:
    """
    Load the list of iteration records from ud.json.
    """
    with ud_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected ud.json format: {ud_json_path}")

    return data


def plot_ud_for_method(
    method: str,
    attempt_name: Optional[str] = None,
    results_root: Optional[Path] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot U(D) search log for a single ICM method.

    默认标题示例: "ICM LL Stub"。
    """
    ud_json_path = get_ud_json_path(
        method=method,
        attempt_name=attempt_name,
        results_root=results_root,
    )
    records = load_ud_records(ud_json_path)

    iterations = [rec["iteration"] for rec in records]
    new_u = [rec.get("new_U") for rec in records]
    cand_u = [rec.get("candidate_U") for rec in records]
    cand_p = [rec.get("candidate_P") for rec in records]
    cand_i = [rec.get("candidate_I") for rec in records]

    if title is None:
        pretty_method = method.replace("_", " ").title()
        title = f"ICM {pretty_method}"

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, new_u, label="new_U")
    plt.plot(iterations, cand_u, label="candidate_U")
    plt.plot(iterations, cand_p, label="candidate_P")
    plt.plot(iterations, cand_i, label="candidate_I")

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()


def discover_methods_for_attempt(
    attempt_name: Optional[str] = None,
    results_root: Optional[Path] = None,
) -> List[str]:
    """
    Discover all methods (subfolders) under `<attempt>/icm`.

    返回方法名列表，例如 ["ll_stub", "official", "utfs"]。
    """
    if results_root is None:
        results_root = get_icm_results_root()

    if attempt_name is None:
        attempt_dir = get_latest_attempt_dir(results_root=results_root)
    else:
        attempt_dir = results_root / attempt_name
        if not attempt_dir.exists():
            raise FileNotFoundError(
                f"Attempt folder not found: {attempt_dir} / 未找到指定 attempt 目录"
            )

    icm_root = attempt_dir / "icm"
    if not icm_root.exists():
        raise FileNotFoundError(
            f"'icm' folder not found under attempt: {icm_root} / 未找到 icm 子目录"
        )

    methods: List[str] = []
    for child in icm_root.iterdir():
        if child.is_dir() and (child / "ud.json").exists():
            methods.append(child.name)

    methods.sort()
    return methods


def plot_ud_for_all_methods(
    attempt_name: Optional[str] = None,
    results_root: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot U(D) search logs for all methods of one attempt.

    默认行为（save_dir=None）：
        - 每个方法的图像保存在它自己的 ud.json 旁边
          (即 <results_root>/attempt_xxx/icm/<method>/<attempt>_<method>_ud.png)

    如果显式提供 save_dir：
        - 全部保存到 save_dir 下。
    """
    if results_root is None:
        results_root = get_icm_results_root()

    if attempt_name is None:
        attempt_dir = get_latest_attempt_dir(results_root=results_root)
        attempt_name = attempt_dir.name

    methods = discover_methods_for_attempt(
        attempt_name=attempt_name,
        results_root=results_root,
    )

    if not methods:
        print(
            f"No methods with ud.json found for attempt {attempt_name} / 未找到任何方法的 ud.json"
        )
        return

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        if save_dir is not None:
            # 集中保存到一个目录
            file_name = f"{attempt_name}_{method}_ud.png"
            save_path = save_dir / file_name
        else:
            # 保存到各自 ud.json 旁边
            ud_json_path = get_ud_json_path(
                method=method,
                attempt_name=attempt_name,
                results_root=results_root,
            )
            file_name = f"{attempt_name}_{method}_ud.png"
            save_path = ud_json_path.parent / file_name

        print(f"Plotting method '{method}' for attempt '{attempt_name}'...")
        plot_ud_for_method(
            method=method,
            attempt_name=attempt_name,
            results_root=results_root,
            save_path=save_path,
            show=show,
        )


# ============================================================
# CLI interface / 命令行入口
# ============================================================


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ICM U(D) search logs (ud.json)."
    )
    parser.add_argument(
        "--attempt",
        type=str,
        default=None,
        help="Name of attempt folder under results (e.g. attempt_20251210_033308). "
        "If omitted, use the latest attempt. / 指定 attempt 目录名，默认使用最新一次。",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="ICM method name under icm/<method> (e.g. ll_stub, official, utfs). "
        "If omitted, plot all methods. / 不指定则对所有方法绘图。",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures instead of (or in addition to) showing them. "
        "/ 将图像保存到文件中。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save figures.\n"
            "- If omitted in single-method mode, save next to ud.json.\n"
            "- If omitted in all-methods mode, save next to each method's ud.json.\n"
            "/ 图像输出目录，默认保存在各自 ud.json 旁边。"
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively. / 不在屏幕上弹出图像窗口。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)

    results_root = get_icm_results_root()
    attempt_name = args.attempt
    show = not args.no_show

    # 默认行为：如果未指定 output-dir，则让保存逻辑在各函数中
    # 决定“各自 ud.json 旁边”的具体路径。
    output_dir: Optional[Path] = None
    if args.save and args.output_dir is not None:
        output_dir = Path(args.output_dir)

    if args.method is not None:
        # 单方法模式
        method = args.method

        # 若设置了 --save 但未指定 output-dir，则在这里解析 ud.json 目录
        # 让图片保存在对应 ud.json 旁边。
        save_path: Optional[Path] = None
        if args.save:
            if output_dir is not None:
                attempt_for_name = (
                    attempt_name
                    if attempt_name is not None
                    else get_latest_attempt_dir(results_root).name
                )
                file_name = f"{attempt_for_name}_{method}_ud.png"
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / file_name
            else:
                ud_json_path = get_ud_json_path(
                    method=method,
                    attempt_name=attempt_name,
                    results_root=results_root,
                )
                attempt_for_name = (
                    attempt_name
                    if attempt_name is not None
                    else ud_json_path.parents[2].name  # attempt_xxx
                )
                file_name = f"{attempt_for_name}_{method}_ud.png"
                save_path = ud_json_path.parent / file_name

        plot_ud_for_method(
            method=method,
            attempt_name=attempt_name,
            results_root=results_root,
            save_path=save_path,
            show=show,
        )
    else:
        # 全方法模式：由 plot_ud_for_all_methods 决定默认路径
        plot_ud_for_all_methods(
            attempt_name=attempt_name,
            results_root=results_root,
            save_dir=output_dir,
            show=show,
        )


if __name__ == "__main__":
    main()
