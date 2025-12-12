# -*- coding: utf-8 -*-
"""
Top-level CLI entry for ICM-tools.
ICM-tools 顶层命令行入口脚本。

This script allows you to:
- Run multiple evaluation settings in one attempt.
- Optionally run a standalone ICM search on the training data.
- Store all evaluation and ICM results under a single attempt folder:
    results/attempt_{timestamp}/
        evaluation/
            ... per-setting JSON + *_arguments.json
        icm/
            official/
            ll_stub/
            utfs/
        output.txt   (captured console output)

该脚本允许你：
- 在一次 attempt 中运行多个评估设置；
- 可选地在训练集上单独运行 ICM；
- 将所有评估与 ICM 结果统一保存在一个 attempt 目录：
    results/attempt_{timestamp}/
        evaluation/
            ... 各 setting 的 JSON + *_arguments.json
        icm/
            official/
            ll_stub/
            utfs/
        output.txt   （整体输出日志）
"""

import os
import sys
import io
import argparse
from datetime import datetime

from eval.evaluation import evaluate
from icm.icm_main import icm_main
from utils.env import get_env_api_key


class Tee(io.TextIOBase):
    """
    Simple tee to write to multiple streams (file + original stdout/stderr).
    将输出同时写入多个流（文件 + 原始 stdout/stderr）的简单 Tee。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            # Force flush immediately to capture logs even if crash happens
            # 立即强制刷新，确保即使发生崩溃也能捕获日志
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _resolve_project_root() -> str:
    """
    Resolve project root as the directory containing this main.py.
    将项目根目录视为 main.py 所在目录。
    """
    return os.path.dirname(os.path.abspath(__file__))


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.
    构建命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="ICM-tools main entry / ICM-tools 顶层入口脚本",
    )

    # What to run / 运行内容选择
    parser.add_argument(
        "--do-eval",
        dest="do_eval",
        action="store_true",
        help="Run evaluation on test data. / 在测试集上运行评估。",
    )
    parser.add_argument(
        "--no-eval",
        dest="do_eval",
        action="store_false",
        help="Do not run evaluation. / 不运行评估。",
    )
    parser.set_defaults(do_eval=True)

    parser.add_argument(
        "--do-icm",
        dest="do_icm",
        action="store_true",
        help=(
            "Also run a standalone ICM search on the training data and save results under icm/. "
            "/ 另外在训练集上单独运行一次 ICM，并将结果保存到 icm/。"
        ),
    )
    parser.set_defaults(do_icm=False)

    # Evaluation-related arguments / 评估相关参数
    parser.add_argument(
        "--eval-settings",
        type=str,
        default="zero_shot,zero_shot_chat,supervised,unsupervised,random_few_shot",
        help=(
            "Comma-separated list of evaluation settings to run, e.g. "
            "zero_shot,unsupervised,random_few_shot "
            "/ 需要运行的评估 setting（逗号分隔）。"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="truthfulqa",
        help="Dataset name tag used in result filenames. / 用于结果文件命名的数据集标签。",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training data JSON (default: project_root/truthfulqa_train.json). "
             "/ 训练集 JSON 路径（默认：项目根目录 truthfulqa_train.json）。",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test data JSON (default: project_root/truthfulqa_test.json). "
             "/ 测试集 JSON 路径（默认：项目根目录 truthfulqa_test.json）。",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B",
        help="Base model name for zero_shot/supervised/unsupervised/random_few_shot. "
             "/ zero_shot / supervised / unsupervised / random_few_shot 使用的 base 模型名称。",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B-Instruct",
        help="Chat/Instruct model name for zero_shot_chat. / zero_shot_chat 使用的 chat/Instruct 模型名称。",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds. / 每次请求的超时时间（秒）。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max new tokens for evaluation completions. / 评估调用生成的新 token 上限。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging. / 启用调试日志。",
    )

    parser.add_argument(
        "--random-fewshot-k",
        type=int,
        default=8,
        help="K for random_few_shot setting. / random_few_shot 模式下 few-shot 示例数量 K。",
    )

    # ICM-related arguments / ICM 相关参数
    parser.add_argument(
        "--icm-mp-method",
        type=str,
        default="official",
        choices=["official", "ll_stub", "utfs"],
        help="Mutual Predictability method for ICM (used for standalone run or single run). / ICM 中使用的 MP 方法（用于独立运行或单次运行）。",
    )
    parser.add_argument(
        "--icm-alpha",
        type=float,
        default=1.0,  # Changed default back to 1.0 as per paper recommendation, user can override.
        help="Alpha in U(D) = alpha * P(D) - I(D). / U(D) 中的 alpha 系数。",
    )
    parser.add_argument(
        "--icm-target-subset-size",
        type=int,
        default=8,
        help="Target subset size K for ICM. / ICM 子集大小 K。",
    )
    parser.add_argument(
        "--icm-max-iter",
        type=int,
        default=256*4,
        help="Max iterations for ICM simulated annealing. / ICM 模拟退火的最大迭代次数。",
    )
    parser.add_argument(
        "--icm-consistency-mode",
        type=str,
        default="at_most_one_true",
        choices=["at_most_one_true", "conflict_count"],
        help="Logical consistency mode for I(D). / I(D) 逻辑一致性模式。",
    )
    parser.add_argument(
        "--icm-enforce-unique-cid",
        action="store_true",
        help="Enforce hard uniqueness per consistency_id. / 对每个 consistency_id 施加硬唯一性约束。",
    )
    parser.add_argument(
        "--icm-initial-t",
        type=float,
        default=5.0,
        help="Initial temperature for ICM. / ICM 初始温度。",
    )
    parser.add_argument(
        "--icm-final-t",
        type=float,
        default=0.1,
        help="Final temperature lower bound for ICM. / ICM 最终温度下限。",
    )
    parser.add_argument(
        "--icm-decay",
        type=float,
        default=0.98,
        help="Decay rate for exponential schedule or factor in log schedule. / 指数或对数温度调度的衰减参数。",
    )
    parser.add_argument(
        "--icm-scheduler",
        type=str,
        default="log",
        choices=["log", "exp"],
        help="Temperature schedule for ICM (log or exp). / ICM 温度调度模式（log 或 exp）。",
    )

    # [NEW] Concurrency Control
    parser.add_argument(
        "--icm-max-concurrent",
        type=int,
        default=4,
        help="Max concurrent API calls for ICM (default 1 to avoid 429). / ICM 最大并发 API 调用数（默认 1 以避免 429）。",
    )

    return parser


def main():
    """
    CLI main function.
    命令行主函数。
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = _resolve_project_root()

    # Default dataset paths if not provided.
    # 若未显式提供，则使用默认的 TruthfulQA 路径。
    if args.train_path is None:
        args.train_path = os.path.join(project_root, "truthfulqa_train.json")
    if args.test_path is None:
        args.test_path = os.path.join(project_root, "truthfulqa_test.json")

    # Create attempt folder and subfolders.
    # 创建 attempt 目录以及 evaluation / icm 子目录。
    attempt_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt_root = os.path.join(project_root, "results", f"attempt_{attempt_timestamp}")
    eval_root = os.path.join(attempt_root, "evaluation")
    icm_root = os.path.join(attempt_root, "icm")
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(icm_root, exist_ok=True)

    # Prepare output.txt tee.
    # 准备 output.txt，拦截并镜像输出。
    output_path = os.path.join(attempt_root, "output.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(output_path, "w", encoding="utf-8") as output_file:
        # Use simple unbuffered Tee
        tee = Tee(original_stdout, output_file)
        sys.stdout = tee
        sys.stderr = tee

        try:
            print("========== ICM-tools main ==========")
            print(f"[MAIN] Project root : {project_root}")
            print(f"[MAIN] Attempt root : {attempt_root}")
            print(f"[MAIN] Evaluation dir: {eval_root}")
            print(f"[MAIN] ICM dir       : {icm_root}")
            print(f"[MAIN] Output log    : {output_path}")
            print(f"[MAIN] Concurrency   : {args.icm_max_concurrent}")
            print("====================================\n")

            api_key = get_env_api_key()

            # --------------------- Run evaluation(s) ---------------------
            # --------------------- 运行评估 ------------------------------
            if args.do_eval:
                raw_settings = [
                    s.strip() for s in args.eval_settings.split(",") if s.strip()
                ]
                # Deduplicate while preserving order.
                # 去重并保持顺序。
                seen = set()
                eval_settings = []
                for s in raw_settings:
                    if s not in seen:
                        seen.add(s)
                        eval_settings.append(s)

                print(f"[MAIN] Evaluation settings: {eval_settings}")

                for setting in eval_settings:
                    if setting == "zero_shot":
                        print("\n[MAIN] Running zero_shot (base model)...")
                        evaluate(
                            data=args.test_path,
                            setting="zero_shot",
                            model=args.base_model,
                            api_key=api_key,
                            train_data_for_icm=None,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                            debug=args.debug,
                            save_result=True,
                            result_root=eval_root,
                            icm_result_root=icm_root,
                            dataset_name=args.dataset_name,
                        )

                    elif setting == "zero_shot_chat":
                        print("\n[MAIN] Running zero_shot_chat (chat model)...")
                        evaluate(
                            data=args.test_path,
                            setting="zero_shot_chat",
                            model=args.chat_model,
                            api_key=api_key,
                            train_data_for_icm=None,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                            debug=args.debug,
                            save_result=True,
                            result_root=eval_root,
                            icm_result_root=icm_root,
                            dataset_name=args.dataset_name,
                        )

                    elif setting == "supervised":
                        print("\n[MAIN] Running supervised (many-shot with gold labels)...")
                        evaluate(
                            data=args.test_path,
                            setting="supervised",
                            model=args.base_model,
                            api_key=api_key,
                            train_data_for_icm=args.train_path,
                            icm_mp_method=args.icm_mp_method,
                            icm_alpha=args.icm_alpha,
                            icm_target_subset_size=args.icm_target_subset_size,
                            icm_max_iter=args.icm_max_iter,
                            icm_consistency_mode=args.icm_consistency_mode,
                            icm_enforce_unique_cid=args.icm_enforce_unique_cid,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                            debug=args.debug,
                            save_result=True,
                            result_root=eval_root,
                            icm_result_root=icm_root,
                            dataset_name=args.dataset_name,
                            random_fewshot_k=args.random_fewshot_k,
                        )

                    elif setting == "unsupervised":
                        # UPDATED LOGIC: Run loop through all methods by default
                        # 更新逻辑：默认循环运行所有方法 (official, ll_stub, utfs)
                        mp_methods = ["official", "ll_stub", "utfs"]
                        print(
                            f"\n[MAIN] Running unsupervised (ICM few-shot) for mp_methods={mp_methods}..."
                        )
                        for mp_method in mp_methods:
                            print(
                                f"\n[MAIN] Running unsupervised (ICM few-shot) with icm_mp_method='{mp_method}'..."
                            )
                            # Create sub-directory logic handled inside evaluate via icm_result_root + mp_method
                            # 子目录逻辑在 evaluate 内部通过 icm_result_root + mp_method 处理

                            evaluate(
                                data=args.test_path,
                                setting=f"unsupervised_{mp_method}",  # Tag setting name
                                model=args.base_model,
                                api_key=api_key,
                                train_data_for_icm=args.train_path,
                                icm_mp_method=mp_method,  # Force current method in loop
                                icm_alpha=args.icm_alpha,
                                icm_target_subset_size=args.icm_target_subset_size,
                                icm_max_iter=args.icm_max_iter,
                                icm_consistency_mode=args.icm_consistency_mode,
                                icm_enforce_unique_cid=args.icm_enforce_unique_cid,
                                icm_max_concurrent=args.icm_max_concurrent,  # Pass concurrency
                                timeout=args.timeout,
                                max_tokens=args.max_tokens,
                                debug=args.debug,
                                save_result=True,
                                result_root=eval_root,
                                icm_result_root=icm_root,
                                dataset_name=args.dataset_name,
                                random_fewshot_k=args.random_fewshot_k,
                            )

                    elif setting == "random_few_shot":
                        print("\n[MAIN] Running random_few_shot (fixed-K gold few-shot)...")
                        evaluate(
                            data=args.test_path,
                            setting="random_few_shot",
                            model=args.base_model,
                            api_key=api_key,
                            train_data_for_icm=args.train_path,
                            icm_mp_method=args.icm_mp_method,
                            icm_alpha=args.icm_alpha,
                            icm_target_subset_size=args.icm_target_subset_size,
                            icm_max_iter=args.icm_max_iter,
                            icm_consistency_mode=args.icm_consistency_mode,
                            icm_enforce_unique_cid=args.icm_enforce_unique_cid,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                            debug=args.debug,
                            save_result=True,
                            result_root=eval_root,
                            icm_result_root=icm_root,
                            dataset_name=args.dataset_name,
                            random_fewshot_k=args.random_fewshot_k,
                        )

                    else:
                        print(
                            f"[MAIN] Unknown setting '{setting}', skipped. / 未知 setting '{setting}'，已跳过。"
                        )

            else:
                print("[MAIN] Evaluation disabled (--no-eval). / 已关闭评估 (--no-eval)。")

            # --------------------- Run standalone ICM ---------------------
            # --------------------- 运行独立 ICM ---------------------------
            if args.do_icm:
                print("\n[MAIN] Running standalone ICM on training data...")
                # Also use specific MP method subfolder for standalone run
                standalone_icm_root = os.path.join(icm_root, args.icm_mp_method)
                os.makedirs(standalone_icm_root, exist_ok=True)

                icm_main(
                    data=args.train_path,
                    model=args.base_model,
                    api_key=api_key,
                    mp_method=args.icm_mp_method,
                    alpha=args.icm_alpha,
                    target_subset_size=args.icm_target_subset_size,
                    max_iter=args.icm_max_iter,
                    initial_t=args.icm_initial_t,
                    final_t=args.icm_final_t,
                    decay=args.icm_decay,
                    scheduler=args.icm_scheduler,
                    use_consistency_term=True,
                    timeout=args.timeout,
                    top_logprobs=20,
                    max_concurrent=args.icm_max_concurrent,  # [Fix] Use CLI arg, not hardcoded 4 / 使用 CLI 参数
                    save_result=True,
                    result_prefix=f"icm_cli_{args.dataset_name}",
                    seed=42,
                    debug=args.debug,
                    consistency_mode=args.icm_consistency_mode,
                    enforce_unique_cid=args.icm_enforce_unique_cid,
                    result_root=standalone_icm_root,
                )
            else:
                print("[MAIN] Standalone ICM disabled (--do-icm not set). / 未启用独立 ICM (--do-icm 未设置)。")

            print("\n[MAIN] All requested work finished. / 所有请求的任务已完成。")

        finally:
            # Restore original stdout/stderr.
            # 恢复原始 stdout/stderr。
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"[MAIN] Full log saved to: {output_path}")


if __name__ == "__main__":
    main()