# -*- coding: utf-8 -*-
"""
Logical consistency scoring helpers.
用于计算逻辑一致性惩罚项 I(D) 的简单规则系统。
"""


class LogicalConsistencyChecker(object):
    """
    Apply a list of logical consistency rules to a dataset.
    对一组样本及对应标签应用若干逻辑一致性规则。
    """

    def __init__(self, rules):
        # rules: iterable of rule objects, each must implement compute_penalty(examples, labels)
        # rules: 可迭代的 rule 对象，每个都应实现 compute_penalty(examples, labels)
        self.rules = list(rules)

    def compute_total_penalty(self, examples, labels):
        """
        Compute the total penalty across all rules.
        计算所有规则下的总惩罚值。
        """
        total = 0
        for rule in self.rules:
            total += rule.compute_penalty(examples, labels)
        return total


class AtMostOneTruePerConsistencyIdRule(object):
    """
    Rule: For each consistency_id, at most one example may have label=True.
    规则：同一个 consistency_id 下，最多只能有一个 label=True。
    """

    def compute_penalty(self, examples, labels):
        from collections import defaultdict

        groups = defaultdict(list)  # consistency_id -> list[example_ids]

        # example = {"id": ..., "question": ..., "choice": ..., "consistency_id": ...}
        # 样本结构示例，至少包括 id 与 consistency_id。
        for ex in examples:
            cid = ex.get("consistency_id")
            ex_id = ex["id"]
            groups[cid].append(ex_id)

        penalty = 0

        for cid, ids in groups.items():
            true_ids = []
            for ex_id in ids:
                if labels.get(ex_id) is True:
                    true_ids.append(ex_id)

            if len(true_ids) > 1:
                # If more than one True, add (len(true_ids) - 1) to penalty.
                # 多于一个 True 则违规，惩罚为 (len(true_ids) - 1)。
                penalty += (len(true_ids) - 1)

        return penalty


class TrueFalseConflictPerConsistencyIdRule(object):
    """
    Rule: For each consistency_id, if both True and False appear, add 1 penalty.
    规则：对每个 consistency_id，如果同时出现 True 和 False，则惩罚加 1。
    """

    def compute_penalty(self, examples, labels):
        from collections import defaultdict

        groups = defaultdict(list)  # consistency_id -> list[example_ids]

        for ex in examples:
            cid = ex.get("consistency_id")
            ex_id = ex["id"]
            groups[cid].append(ex_id)

        penalty = 0

        for cid, ids in groups.items():
            has_true = False
            has_false = False

            for ex_id in ids:
                val = labels.get(ex_id, False)
                if val is True:
                    has_true = True
                else:
                    # We treat missing or False as False for conflict counting.
                    # 这里将缺失或 False 视为 False，用于判断是否与 True 冲突。
                    has_false = True

            if has_true and has_false:
                penalty += 1

        return penalty


def score_logical_consistency(examples, labels, rules=None, mode: str = "at_most_one_true"):
    """
    Compute total logical consistency penalty.
    使用一组规则计算逻辑一致性的总惩罚值。

    Args / 参数:
        examples:
            List/iterable of examples. Each example must have:
                - "id"
                - "consistency_id"
            and can contain other fields.
            样本列表/可迭代对象，每个样本至少包含 "id" 和 "consistency_id" 字段。

        labels:
            dict: id -> bool (True means "labelled as true", False otherwise)
            字典：id -> bool，True 表示“认为为真”，False 表示“认为为假”。

        rules:
            Optional list of rule instances (overrides `mode` if provided).
            可选的规则实例列表（如果显式传入则覆盖 mode 所指定的默认规则）。

        mode:
            Which predefined rule set to use when rules is None.
            当 rules 为 None 时，选择使用哪一套预定义规则。

            - "at_most_one_true":
                Use AtMostOneTruePerConsistencyIdRule.
                使用 AtMostOneTruePerConsistencyIdRule（同一组 True 超过一个就加罚）。

            - "conflict_count":
                Use TrueFalseConflictPerConsistencyIdRule.
                使用 TrueFalseConflictPerConsistencyIdRule（同一组内 True/False 同时出现则记一次冲突）。

    Returns / 返回:
        float: total penalty value (larger => more inconsistent).
        float: 总惩罚值（越大表示逻辑不一致程度越高）。
    """
    if rules is None:
        if mode == "at_most_one_true":
            rules = [AtMostOneTruePerConsistencyIdRule()]
        elif mode == "conflict_count":
            rules = [TrueFalseConflictPerConsistencyIdRule()]
        else:
            raise ValueError(
                'Unknown logical consistency mode: {m}. '
                'Expected "at_most_one_true" or "conflict_count". / '
                '未知的逻辑一致性模式，期望 "at_most_one_true" 或 "conflict_count".'
                .format(m=mode)
            )

    checker = LogicalConsistencyChecker(rules)
    penalty = checker.compute_total_penalty(examples, labels)
    return float(penalty)


async def score_logical_consistency_async(examples, labels, rules=None, mode: str = "at_most_one_true"):
    """
    Async wrapper for score_logical_consistency.
    score_logical_consistency 的异步包装，方便与其他 async 代码组合使用。
    """
    return score_logical_consistency(examples, labels, rules=rules, mode=mode)
