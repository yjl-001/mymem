# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re


GSM8K_VERIFIER_VERSION = "gsm8k-first-boxed-v2-diagnostic"


_NUMERIC_ANSWER_RE = re.compile(
    r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:%|\\%)?(?!\w)"
)


def compute_score(completion, ground_truth) -> float:
    retval = 0.0
    try:
        # string_in_last_boxed = last_boxed_only_string(solution_str)
        string_in_first_boxed = first_boxed_only_string(completion)
        ground_truth_in_last_boxed = last_boxed_only_string(ground_truth)
        if string_in_first_boxed is not None:
            answer = remove_boxed(string_in_first_boxed)
            ground_truth = remove_boxed(ground_truth_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


def extract_last_numeric_answer(text):
    """Best-effort diagnostic extraction; never changes the official reward.

    GSM8K's task contract requires a boxed answer, so ``compute_score`` remains
    strict.  This extractor only helps distinguish a format-only failure from a
    likely answer failure when a rollout omitted or malformed the box.
    """

    matches = _NUMERIC_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def diagnose_gsm8k_completion(completion, ground_truth):
    """Return a structured diagnosis alongside the strict GSM8K reward."""

    reward = float(compute_score(completion, ground_truth))
    predicted_box = first_boxed_only_string(completion)
    expected_box = last_boxed_only_string(ground_truth)
    try:
        predicted_answer = (
            remove_boxed(predicted_box) if predicted_box is not None else None
        )
    except (TypeError, ValueError):
        predicted_answer = None
    try:
        expected_answer = remove_boxed(expected_box) if expected_box is not None else None
    except (TypeError, ValueError):
        expected_answer = None
    box_marker_present = "\\boxed" in completion or "\\fbox" in completion
    format_valid = predicted_box is not None and predicted_answer is not None

    if predicted_answer is not None:
        diagnostic_answer = predicted_answer
        diagnostic_source = "first_boxed_answer"
    else:
        diagnostic_answer = extract_last_numeric_answer(completion)
        diagnostic_source = (
            "last_numeric_candidate" if diagnostic_answer is not None else "unavailable"
        )

    diagnostic_answer_correct = None
    if diagnostic_answer is not None and expected_answer is not None:
        diagnostic_answer_correct = bool(is_equiv(diagnostic_answer, expected_answer))

    failure_types = []
    if reward == 0.0:
        if not format_valid:
            failure_types.append(
                "malformed_boxed" if box_marker_present else "missing_boxed"
            )
            if diagnostic_answer_correct is False:
                failure_types.append("answer_mismatch")
            elif diagnostic_answer_correct is None:
                failure_types.append("answer_unverified")
        else:
            failure_types.append("boxed_answer_mismatch")

    return {
        "version": GSM8K_VERIFIER_VERSION,
        "reward": reward,
        "task_success": reward == 1.0,
        "format_required": True,
        "format_valid": format_valid,
        "predicted_answer": predicted_answer,
        "expected_answer": expected_answer,
        "diagnostic_answer": diagnostic_answer,
        "diagnostic_answer_source": diagnostic_source,
        "diagnostic_answer_correct": diagnostic_answer_correct,
        "failure_types": failure_types,
    }


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if not isinstance(s, str):
        raise TypeError("boxed expression must be a string")
    for command in ("\\boxed", "\\fbox"):
        spaced_prefix = command + " "
        if s.startswith(spaced_prefix):
            return s[len(spaced_prefix) :]
        braced_prefix = command + "{"
        if s.startswith(braced_prefix) and s.endswith("}"):
            return s[len(braced_prefix) : -1]
    raise ValueError("unsupported or malformed boxed expression")


def first_boxed_only_string(string):
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[1].split("$")[0]

    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove escaped or raw currency delimiters inside a valid answer box
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
