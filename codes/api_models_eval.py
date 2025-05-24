#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call GPT-4o on every answer‑pair record and save full model responses to Excel, including
an extra column with the last line of each model response.

Usage
-----
python evaluate_pairs_to_xlsx_v2.py <language>
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import openai
from openpyxl import Workbook

# openai.api_key = "your_api_key"  # TODO: replace with real key or env var
# openai.base_url = "your_base_url"  # openai==1.x
openai.api_key = "sk-fa663c1cb5594070b258310e4ba99d40"
openai.base_url = "https://api.deepseek.com"   
MODEL_NAME = "deepseek-reasoner"


def build_prompt(
    question: str,
    reference: str,
    ans1: str,
    ans2: str,
    language: str
) -> str:
    """Construct the evaluation prompt fed to the GPT model based on the language."""
    if language == 'zh':
        return f"""现有如下问题：

问题：{question}
该问题的**标准答案**如下：{reference}

我们现在得到了两份学生对该问题的回答。
回答一的内容如下：{ans1}
回答二的内容如下：{ans2}

现在，你应当以该问题的内容为上下文背景，必须以标准答案为判决依据，判决答案一和答案二中哪份更贴近标准答案的内容。
现在，请你先展开简单的分析，随后给出结论：回答一或回答二更优秀
有时，回答一与回答二和标准答案相比也有可能不相上下。

你的回答必须在最后一行以如下形式给出：
因此，【回答一】更优秀 或 因此，【回答二】更优秀
若你认为这两个回答不相上下，你应当在最后一行以如下形式给出：
因此，【两个回答相同】
"""
    else:
        # Default to English prompt
        return f"""We have the following question:

Question: {question}
The reference (standard) answer to this question is as follows: {reference}

We now have two student responses to this question.
Response 1 is as follows: {ans1}
Response 2 is as follows: {ans2}

Now, you should evaluate the two responses based on the content of the question, using the standard answer as the sole basis for judgment.
Determine which of the two—Response 1 or Response 2—is closer to the reference answer in terms of content.

Please begin with a brief analysis, and then provide your final judgment in one of the following forms:

If one response is better:
"Therefore, [Response 1] is better." or "Therefore, [Response 2] is better."

If the two responses are roughly equal in quality:
"Therefore, [Both responses are equal]."""  # noqa: E501


def call_gpt(prompt: str) -> str:
    """Send the prompt to the GPT model and return its reply."""
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


PAIR_KEYS = (
    "student_answer_a_vs_student_answer_b",
    "student_answer_a_vs_model_answer_a",
    "student_answer_a_vs_model_answer_b",
    "student_answer_b_vs_model_answer_a",
    "student_answer_b_vs_model_answer_b",
    "model_answer_a_vs_model_answer_b",
)


def extract_pair_and_fields(item: dict) -> Tuple[str, str, str]:
    """Return the pair key and the corresponding field names for the left and right answers."""
    for key in PAIR_KEYS:
        if key in item:
            left_base, right_base = key.split("_vs_")
            return key, f"{left_base}", f"{right_base}"
    raise ValueError("cannot find pair key!")


def main(language: str) -> None:
    # Determine the input JSON and output Excel paths based on the language
    if language == 'zh':
        input_json = Path("../data/LFQA-E-zh.json").resolve()
        output_xlsx = Path(f"../results/{MODEL_NAME}_zh.xlsx").resolve()
    else:
        input_json = Path("../data/LFQA-E-en.json").resolve()
        output_xlsx = Path(f"../results/{MODEL_NAME}_en.xlsx").resolve()

    items = json.loads(input_json.read_text(encoding="utf-8"))

    wb = Workbook()
    ws = wb.active
    ws.title = "full_model_responses"
    # ➕ Add an extra column header for the last line of the model response
    ws.append([
        "pair_type",
        "model_response",
        "human_label_raw",
        "model_response_verdict",
    ])

    for idx, item in enumerate(items, 1):
        pair_key, left_field, right_field = extract_pair_and_fields(item)

        question = item["question"]
        reference = item.get("Concise_Reference") or item.get("Concise_Reference", "")
        ans1 = item.get(left_field, "")
        ans2 = item.get(right_field, "")
        human_raw = item[pair_key]

        prompt = build_prompt(question, reference, ans1, ans2, language)

        try:
            gpt_reply = call_gpt(prompt)
        except Exception as e:
            print(f"[{idx}] invoke failed: {e}")
            gpt_reply = f"ERROR: {e}"

        # Extract the last non‑empty line from the model response
        last_line = gpt_reply.splitlines()[-1].strip() if gpt_reply else ""

        # Write a row including the extra column
        ws.append([pair_key, gpt_reply, human_raw, last_line])
        print(f"[{idx:>4}/{len(items)}] {pair_key} → finished")

    wb.save(output_xlsx)
    print(f"\nAll finished, results saved to {output_xlsx}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluate_pairs_to_xlsx_v2.py <language>")
        print("language should be 'zh' for Chinese or 'en' for English.")
        sys.exit(1)

    language = sys.argv[1].lower()

    if language not in ['zh', 'en']:
        print("Error: language should be 'zh' for Chinese or 'en' for English.")
        sys.exit(1)

    main(language)
