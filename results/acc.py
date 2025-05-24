#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate GPT‑4o pair‑wise judgments stored in the *NEW* Excel layout produced by
`evaluate_pairs_to_xlsx_v2_modified.py`.

The input sheet must contain these columns:
    • pair_type               —— specifies which of the six comparison types the row belongs to
    • model_response_last_line —— model's final verdict sentence (previously {prefix}_model_verdict)
    • human_label_raw         —— human gold label (previously {prefix}_human)

The script prints exactly the same per‑category / overall accuracy & F1 report
as the original version.

Usage
-----
python evaluate_accuracy_with_f1_modified.py <language> results_part1.xlsx [more.xlsx ...]
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# —— 与生成脚本保持一致的 6 组比对关系 ——
PAIR_KEYS = [
    ("student_answer_a", "student_answer_b", "student_answer_a_vs_student_answer_b"),
    ("student_answer_a", "model_answer_a", "student_answer_a_vs_model_answer_a"),
    ("student_answer_a", "model_answer_b", "student_answer_a_vs_model_answer_b"),
    ("student_answer_b", "model_answer_a", "student_answer_b_vs_model_answer_a"),
    ("student_answer_b", "model_answer_b", "student_answer_b_vs_model_answer_b"),
    ("model_answer_a", "model_answer_b", "model_answer_a_vs_model_answer_b"),
]

# —— 建立 prefix → (left_answer_field, right_answer_field) 的映射 ——
PREFIX2KEYS = {prefix: (k1, k2) for k1, k2, prefix in PAIR_KEYS}

# ———— 归一化工具 ————

def canon_model(txt: str) -> str:
    """模型判决 → resp1 / resp2 / equal / unknown"""
    t = txt.lower()
    if "【两个回答相同】" in t or "两个回答相同" in t or "回答相同" in t:
        return "equal"
    if "【回答一】" in t or "**回答一**" in t:
        return "resp1"
    if "【回答二】" in t or "**回答二**" in t:
        return "resp2"
    return "unknown"

def canon_human(txt: str, k1: str, k2: str) -> str:
    """人工标签 → resp1 / resp2 / equal / unknown"""
    t = txt.lower()
    if "equal" in t or "same" in t:
        return "equal"
    # 用左、右答案在列名中的缩写来匹配
    if k1.split("_vs_")[0] in t:
        return "resp1"
    if k2.split("_vs_")[0] in t:
        return "resp2"
    return "unknown"



# ———— F1 计算函数 ————

def calculate_f1(stats: dict) -> dict:
    classes = ["resp1", "resp2", "equal"]
    metrics = {}

    # 每个类别的 P / R / F1
    for cls in classes:
        tp = stats[f"tp_{cls}"]
        fp = stats[f"fp_{cls}"]
        fn = stats[f"fn_{cls}"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics[f"precision_{cls}"] = precision
        metrics[f"recall_{cls}"] = recall
        metrics[f"f1_{cls}"] = f1

    # 宏观平均
    metrics["macro_f1"] = np.mean([metrics[f"f1_{cls}"] for cls in classes])

    # 微观平均
    total_tp = sum(stats[f"tp_{cls}"] for cls in classes)
    total_fp = sum(stats[f"fp_{cls}"] for cls in classes)
    total_fn = sum(stats[f"fn_{cls}"] for cls in classes)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    metrics["micro_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    return metrics

# ———— 主评估函数 ————

def evaluate_one_file(path: Path, totals: defaultdict, unknown_cases: list, language: str) -> None:
    df = pd.read_excel(path)

    required_cols = {"pair_type", "model_response_verdict", "human_label_raw"}
    if not required_cols.issubset(df.columns):
        print(f"[WARN] {path.name} skipped: missing required columns {required_cols - set(df.columns)}")
        return

    # Excel 中 1‑based 列索引
    mcol_idx = df.columns.get_loc("model_response_verdict") + 1
    hcol_idx = df.columns.get_loc("human_label_raw") + 1

    for ridx, row in df.iterrows():
        prefix = str(row["pair_type"]).strip()
        if prefix not in PREFIX2KEYS:  # 非六大类别 → 跳过
            continue

        k1, k2 = PREFIX2KEYS[prefix]
        model_txt = str(row["model_response_verdict"])
        human_txt = str(row["human_label_raw"])

        model = canon_model(model_txt)
        human = canon_human(human_txt, k1, k2)
        excel_row = ridx + 2  # +1 header, +1 0‑base → 1‑base

        # —— 记录 unknown 情况 ——
        if model == "unknown":
            unknown_cases.append((path.name, excel_row, mcol_idx))
        if human == "unknown":
            unknown_cases.append((path.name, excel_row, hcol_idx))

        # —— 无人工参考 → 不计入指标 ——
        if human == "unknown":
            continue

        totals[prefix]["total"] += 1
        if model == human:
            totals[prefix]["correct"] += 1

        # TP / FP / FN 更新
        classes = ["resp1", "resp2", "equal"]
        for cls in classes:
            if human == cls and model == cls:
                totals[prefix][f"tp_{cls}"] += 1
            elif human != cls and model == cls:
                totals[prefix][f"fp_{cls}"] += 1
            elif human == cls and model != cls:
                totals[prefix][f"fn_{cls}"] += 1

    if language == 'zh':
        print("\n=== 分类指标 ===")
    else:
        print("\n=== Per-category metrics ===")

def print_results(totals, grand_correct, grand_total, grand_tp, grand_fp, grand_fn, classes, language):
    if language == 'zh':
        print(f"总体准确率: {grand_correct}/{grand_total} = {grand_correct / grand_total:.2%}")
    else:
        print(f"Overall accuracy: {grand_correct}/{grand_total} = {grand_correct / grand_total:.2%}")

    # 输出宏观平均值和微观平均值
    grand_stats = {}
    for cls in classes:
        grand_stats[f"tp_{cls}"] = grand_tp[cls]
        grand_stats[f"fp_{cls}"] = grand_fp[cls]
        grand_stats[f"fn_{cls}"] = grand_fn[cls]

    grand_f1 = calculate_f1(grand_stats)
    if language == 'zh':
        print(f"总体宏观 F1: {grand_f1['macro_f1']:.4f}")
        print(f"总体微观 F1: {grand_f1['micro_f1']:.4f}")
        print("\n各类 F1 分数：")
    else:
        print(f"Overall Macro F1: {grand_f1['macro_f1']:.4f}")
        print(f"Overall Micro F1: {grand_f1['micro_f1']:.4f}")
        print("\nClass F1 scores across all categories:")

    for cls in classes:
        print(
            f"  - {cls}: {grand_f1[f'f1_'+cls]:.4f} "
            f"(P={grand_f1[f'precision_'+cls]:.4f}, R={grand_f1[f'recall_'+cls]:.4f})"
        )

def main(files, language):
    totals = defaultdict(lambda: defaultdict(int))
    unknown_cases = []

    for fp in files:
        evaluate_one_file(Path(fp), totals, unknown_cases, language)

    # —— 输出结果 ——
    grand_correct = grand_total = 0
    grand_tp = defaultdict(int)
    grand_fp = defaultdict(int)
    grand_fn = defaultdict(int)
    classes = ["resp1", "resp2", "equal"]

    print_results(totals, grand_correct, grand_total, grand_tp, grand_fp, grand_fn, classes, language)

    # —— 输出 unknown 单元格 ——
    if unknown_cases:
        if language == 'zh':
            print("\n=== 无法判定的单元格 ===")
        else:
            print("\n=== Unable-to-judge cells ===")
        for fname, r, c in unknown_cases:
            print(f"{fname:<30s}  Row {r:<5d}  Col {c}")
    else:
        if language == 'zh':
            print("\n没有无法判定的单元格。")
        else:
            print("\nNo unknown cases.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("用法: python evaluate_accuracy_with_f1_modified.py <language> results_part1.xlsx [more.xlsx ...]")
    language = sys.argv[1].lower()
    if language not in ['zh', 'en']:
        sys.exit("语言参数应为 'zh' 或 'en'")

    main(sys.argv[2:], language)
