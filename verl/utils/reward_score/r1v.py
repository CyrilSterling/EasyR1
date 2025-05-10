# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict

from math_verify import parse, verify
from mathruler.grader import extract_boxed_content
from sympy import pi


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def fix_frac(expr: str) -> str:
    # frac{xxx}{xxx} -> \frac{xxx}{xxx}
    expr = re.sub(r"(?<!\\)frac", r"\\frac", expr)
    # \fracab, \frac{a}b, \fraca{b}, \frac(a)b, \fraca(b), \frac(a)(b) -> \frac{a}{b}
    expr = re.sub(r"\\frac([^{\s])([^{\s])", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac(\{[^{}]+\})([^{\s])", r"\\frac\1{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])(\{[^{}]+\})", r"\\frac{\1}\2", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)([^{\s])", r"\\frac{\1}{\2}", expr)
    return expr


def fix_sqrt(expr: str) -> str:
    # sqrt{xxx} -> \sqrt{xxx}
    expr = re.sub(r"(?<!\\)sqrt", r"\\sqrt", expr)
    # \sqrt(xxx) -> \sqrt{xxx}
    expr = re.sub(r"\\sqrt\((.*?)\)", r"\\sqrt{\1}", expr)
    # \sqrtxxxx -> \sqrt{x}xxx
    expr = re.sub(r"\\sqrt(?!\{)(.)", r"\\sqrt{\1}", expr)
    return expr


def fix_pi(expr: str) -> str:
    # pi -> \pi
    expr = re.sub(r"(?<!\\)pi", r"\\pi", expr)
    expr = expr.replace("π", "\\pi")
    return expr


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_decimal_places(s):
    match = re.search(r"\.(\d+)", s)
    return len(match.group(1)) if match else 0


def replace_circled_numbers(text: str) -> str:
    def circled_to_digit(match):
        char = match.group(0)
        return str(ord(char) - 0x2460 + 1)

    pattern = r"[\u2460-\u2473]"
    if re.search(pattern, text) is not None:
        text = text.replace(",", "")
        return re.sub(pattern, circled_to_digit, text)
    return text


def normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    # m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    # if m is not None:
    #     expr = m.group("text")
    # Remove enclosing `\text{}`. Execute twice to account for two levels of nesting.
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "dm",
        "meter",
        "mile",
        "gram",
        "kilo",
        "kilogram",
        "kg",
        "liter",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "in",
        "yard",
        "square",
        "cell",
        "unit",
        "yuan",
        "time",
        "米",
        "厘米",
        "克",
        "千克",
        "公斤",
        "升",
        "秒",
        "分钟",
        "分",
        "小时",
        "天",
        "周",
        "月",
        "年",
        "元",
    ]:
        # end by es/s, ^d or unicode superscript
        expr = re.sub(
            rf"{unit}(es)?(s)? *(\^({{*)[0-9]+(}}*))?([\u00B2\u00B3\u2070-\u2079]+)?",
            "",
            expr,
        )

    # delete \cric or ^\cric or ^{\cric} or unicode format degree
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = re.sub(r"\^ *{\\circ}", "", expr)
    expr = re.sub(r"\\circ", "", expr)
    expr = re.sub(r"°", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    # if _is_float(expr) and _is_int(float(expr)):
    #     expr = str(int(round(float(expr))))
    # if "\\" in expr:
    #     try:
    #         expr = _parse_latex(expr)
    #     except:
    #         pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    # expr = expr.replace("{", "")
    # expr = expr.replace("}", "")

    # if we somehow still have blank (), drop them
    expr = expr.replace("()", "")
    expr = expr.replace("{}", "")

    expr = expr.replace("√", "\\sqrt")
    expr = fix_frac(expr)
    expr = fix_sqrt(expr)
    expr = fix_pi(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    # if _str_is_int(expr):
    #     expr = str(_str_to_int(expr))
    # Geometry
    expr = expr.replace("\\parallel", "//")
    expr = expr.replace("平行", "//")
    expr = expr.replace("⊥", "\\perp")
    expr = expr.replace("△", "\\triangle")
    expr = expr.replace("Δ", "\\triangle")
    expr = expr.replace("∠", "\\angle")
    expr = expr.replace("∽", "\\sim")
    expr = expr.replace("角", "\\angle")
    expr = expr.replace("平面", "plane")
    expr = expr.replace("且", "and")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("正确", "correct")
    expr = expr.replace("错误", "incorrect")
    expr = expr.replace("notlessthan", "\\geq")
    expr = expr.replace("notmorethan", "\\leq")
    expr = replace_circled_numbers(expr)
    if "不够" in expr:
        expr = "no"
    elif "够" in expr:
        expr = "yes"
    if "notenough" in expr:
        expr = "no"
    elif "enough" in expr:
        expr = "yes"
    if "not" in expr:
        expr = "no"

    return expr


def is_choice_format(expr):
    expr = expr.strip().upper()
    return bool(re.match(r"^([A-E]$|\([A-E]\)$|[A-E]\.|\([A-E]\))", expr))


def r1v_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0
    # if format_match:
    #     content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL).group(1).strip()
    #     boxed_pattern = re.compile(r"\\boxed\{.*?\}", re.DOTALL)
    #     boxed_match = re.search(boxed_pattern, content_match)
    #     if boxed_match is not None:
    #         return 1.0
    # return 0.0


def r1v_accuracy_reward(
    predict_str: str, ground_truth: str, response_length=None
) -> float:
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        pred_answer = content_match.group(
            1
        ).strip()  # if content_match else predict_str.strip()
        pred_answer = extract_boxed_content(pred_answer).strip()
        ## original mathruler match code
        # # pred_answer = extract_boxed_content(pred_answer)
        if is_choice_format(pred_answer) or is_choice_format(ground_truth):
            pattern = r"^\(?([A-E])\)?(?:\.\s*|$|\s)"
            pred_answer = re.match(pattern, pred_answer.strip().upper()).group(1)
            ground_truth = re.match(pattern, ground_truth.strip().upper()).group(1)
            if pred_answer == ground_truth:
                return 1.0
            else:
                return 0.1
        if pred_answer == ground_truth:
            return 1.0
        pred_answer = normalize(pred_answer).strip()
        ground_truth = normalize(ground_truth).strip()
        if pred_answer == ground_truth:
            return 1.0

        if _is_float(pred_answer) and _is_float(ground_truth):
            float_rounding_limit = min(
                len(pred_answer.split(".")[-1]), len(ground_truth.split(".")[-1])
            )
        elif "pi" in pred_answer or "pi" in ground_truth:
            float_rounding_limit = 2
        else:
            float_rounding_limit = 4
        pred_answer = parse(f"\\boxed{{{pred_answer}}}")
        ground_truth = parse(f"\\boxed{{{ground_truth}}}")

        # consider the constant pi
        pred_answer[0] = pred_answer[0].subs(pi, 3.14)
        ground_truth[0] = ground_truth[0].subs(pi, 3.14)

        # if content_match is None or pred_answer is None:
        #     return 0.0
        # print(pred_answer, ground_truth)
        if verify(pred_answer, ground_truth, float_rounding=float_rounding_limit):
            return 1.0
        else:
            return 0.1
    except Exception as e:
        if isinstance(e, ValueError):
            print(
                f"Error occurred when computing reward!\n[[predict_str]] {predict_str}\n[[ground_truth]] {ground_truth}"
            )
        return 0.0


def r1v_accuracy_only_reward(
    predict_str: str, ground_truth: str, response_length=None
) -> float:
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        pred_answer = content_match.group(
            1
        ).strip()  # if content_match else predict_str.strip()
        pred_answer = extract_boxed_content(pred_answer).strip()
        ## original mathruler match code
        # # pred_answer = extract_boxed_content(pred_answer)
        if is_choice_format(pred_answer) or is_choice_format(ground_truth):
            pattern = r"^\(?([A-E])\)?(?:\.\s*|$|\s)"
            pred_answer = re.match(pattern, pred_answer.strip().upper()).group(1)
            ground_truth = re.match(pattern, ground_truth.strip().upper()).group(1)
            if pred_answer == ground_truth:
                return 1.0
        if pred_answer == ground_truth:
            return 1.0
        pred_answer = normalize(pred_answer).strip()
        ground_truth = normalize(ground_truth).strip()
        if pred_answer == ground_truth:
            return 1.0

        if _is_float(pred_answer) and _is_float(ground_truth):
            float_rounding_limit = min(
                len(pred_answer.split(".")[-1]), len(ground_truth.split(".")[-1])
            )
        elif "pi" in pred_answer or "pi" in ground_truth:
            float_rounding_limit = 2
        else:
            float_rounding_limit = 4
        pred_answer = parse(f"\\boxed{{{pred_answer}}}")
        ground_truth = parse(f"\\boxed{{{ground_truth}}}")

        # consider the constant pi
        pred_answer[0] = pred_answer[0].subs(pi, 3.14)
        ground_truth[0] = ground_truth[0].subs(pi, 3.14)

        # if content_match is None or pred_answer is None:
        #     return 0.0
        # print(pred_answer, ground_truth)
        if verify(pred_answer, ground_truth, float_rounding=float_rounding_limit):
            return 1.0
    except Exception as e:
        if isinstance(e, ValueError):
            print(
                f"Error occurred when computing reward!\n[[predict_str]] {predict_str}\n[[ground_truth]] {ground_truth}"
            )
        return 0.0

    return 0.0


def r1v_compute_score_(
    predict_str: str, ground_truth: str, validation: bool = False, response_length=None
) -> float:
    acc_reward = r1v_accuracy_reward(predict_str, ground_truth, response_length)
    format_reward = r1v_format_reward(predict_str)
    if validation:
        reward = acc_reward if acc_reward == 1.0 else 0.0
    else:
        train_acc_reward = acc_reward
        reward = train_acc_reward * 0.9 + format_reward * 0.1
    # reward /= 2
    return reward


def r1v_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format = r1v_format_reward(predict_str)
    accuracy = r1v_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * accuracy + 0.5 * format,
        "format": format,
        "accuracy": accuracy,
    }
