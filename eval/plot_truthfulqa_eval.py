# -*- coding: utf-8 -*-
"""
TruthfulQA evaluation bar plots.
自动绘制两幅柱状图，并保存到 attempt/evaluation/ 内。

Figure 1:
    Zero-shot, Zero-shot (Chat), Supervised, Unsupervised (official)

Figure 2:
    Zero-shot, Zero-shot (Chat),
    Supervised, Random few-shot,
    Unsupervised (official, utfs, ll_stub)
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Path resolution helpers
# ============================================================

def get_project_root() -> Path:
    """
    Assume current file is at project_root/eval/plot_truthfulqa_eval.py
    """
    return Path(__file__).resolve().parent.parent


def get_results_root() -> Path:
    return get_project_root() / "results"


def list_attempt_dirs(results_root: Optional[Path] = None) -> List[Path]:
    if results_root is None:
        results_root = get_results_root()

    attempt_dirs = [
        d for d in results_root.iterdir()
        if d.is_dir() and d.name.startswith("attempt_")
    ]
    attempt_dirs.sort(key=lambda p: p.name)
    return attempt_dirs


def get_latest_attempt_dir(results_root: Optional[Path] = None) -> Path:
    attempts = list_attempt_dirs(results_root)
    if not attempts:
        raise RuntimeError("No attempt_* folders found under results/")
    return attempts[-1]


def get_evaluation_dir(
    attempt_name: Optional[str] = None,
    results_root: Optional[Path] = None,
) -> Path:

    if results_root is None:
        results_root = get_results_root()

    if attempt_name is None:
        attempt_dir = get_latest_attempt_dir(results_root)
    else:
        attempt_dir = results_root / attempt_name
        if not attempt_dir.exists():
            raise FileNotFoundError(f"Attempt folder not found: {attempt_dir}")

    eval_dir = attempt_dir / "evaluation"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation folder missing: {eval_dir}")

    return eval_dir


# ============================================================
# JSON loading helpers
# ============================================================

def make_sorted_accuracy_list(
    eval_dir: Path,
    name_file_pairs: List[Tuple[str, str]],
) -> List[Tuple[str, float]]:
    """
    对给定的 (显示名, 文件名) 列表：
    - 如果文件存在：读取 accuracy_on_parsed * 100
    - 如果不存在：warning + 忽略
    - 最终按 accuracy 升序排序
    """
    results: List[Tuple[str, float]] = []

    for display_name, filename in name_file_pairs:
        path = eval_dir / filename
        if not path.exists():
            print(f"[WARN] Missing summary file, skip: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        acc = float(data["accuracy_on_parsed"]) * 100.0
        results.append((display_name, acc))

    if not results:
        raise RuntimeError(
            f"No valid summary files found under {eval_dir} for this figure."
        )

    return sorted(results, key=lambda x: x[1])


# ============================================================
# Plot helper (dynamic width)
# ============================================================

def _plot_bar(
    results: List[Tuple[str, float]],
    title: str,
    save_path: Path,
    show: bool = True,
    ylim=(30, 100),
) -> None:

    labels = [x[0] for x in results]
    values = [x[1] for x in results]

    # 动态宽度：每个柱子 1.3 inch，最少 8 inch
    fig_width = max(8, 1.3 * len(labels))
    plt.figure(figsize=(fig_width, 6))

    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("accuracy (%)")

    # 长标题自动换行
    plt.title(title, wrap=True)

    plt.ylim(*ylim)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Figure 1
# ============================================================

def plot_truthfulqa_main_settings(
    attempt_name: Optional[str] = None,
    show: bool = True,
) -> Path:

    eval_dir = get_evaluation_dir(attempt_name)

    name_file_pairs = [
        ("Zero-shot", "zero_shot_truthfulqa_summary.json"),
        ("Zero-shot (Chat)", "zero_shot_chat_truthfulqa_summary.json"),
        ("Supervised", "supervised_truthfulqa_summary.json"),
        ("Unsupervised (official)", "unsupervised_official_truthfulqa_summary.json"),
    ]

    results = make_sorted_accuracy_list(eval_dir, name_file_pairs)

    save_path = eval_dir / "truthfulqa_main_settings_bar.png"
    _plot_bar(
        results,
        title="TruthfulQA: Zero-shot / Supervised / Unsupervised (official)",
        save_path=save_path,
        show=show,
    )

    return save_path


# ============================================================
# Figure 2
# ============================================================

def plot_truthfulqa_unsup_variants(
    attempt_name: Optional[str] = None,
    show: bool = True,
) -> Path:

    eval_dir = get_evaluation_dir(attempt_name)

    name_file_pairs = [
        ("Zero-shot", "zero_shot_truthfulqa_summary.json"),
        ("Zero-shot (Chat)", "zero_shot_chat_truthfulqa_summary.json"),
        ("Supervised", "supervised_truthfulqa_summary.json"),
        ("Random few-shot", "random_few_shot_truthfulqa_summary.json"),
        ("Unsupervised (official)", "unsupervised_official_truthfulqa_summary.json"),
        ("Unsupervised (utfs)", "unsupervised_utfs_truthfulqa_summary.json"),
        ("Unsupervised (ll stub)", "unsupervised_ll_stub_truthfulqa_summary.json"),
    ]

    results = make_sorted_accuracy_list(eval_dir, name_file_pairs)

    save_path = eval_dir / "truthfulqa_unsup_variants_bar.png"
    _plot_bar(
        results,
        title=("TruthfulQA: Zero-shot / Supervised / Random Few-shot / "
               "Unsupervised Variants"),
        save_path=save_path,
        show=show,
    )

    return save_path


# ============================================================
# CLI
# ============================================================

def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    show = not args.no_show

    p1 = plot_truthfulqa_main_settings(args.attempt, show=show)
    p2 = plot_truthfulqa_unsup_variants(args.attempt, show=show)

    print("Generated figures:")
    print(" -", p1)
    print(" -", p2)


if __name__ == "__main__":
    main()
