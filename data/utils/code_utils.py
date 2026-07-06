#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Any, Optional
import re
import multiprocessing
from multiprocessing.connection import Connection

ExecuteResult = Tuple[bool, str, Tuple[bool]]


def extract_python_code(text_string: str) -> List[str]:
    code_blocks = re.findall(r"```python(.*?)```", text_string, re.DOTALL)
    if not code_blocks:
        code_blocks = [text_string]

    results = []
    for block in code_blocks:
        imports = re.findall(r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+.*)$", block, re.MULTILINE)

        funcs = re.findall(r"(def\s+\w+\(.*?:[\s\S]*?)(?=^def\s|\Z)", block.strip(), re.MULTILINE)
        funcs = [_trim_to_function(f) for f in funcs]

        if imports:
            import_block = "\n".join(imports)
            if funcs:
                funcs = [import_block] + funcs
            else:
                funcs = [import_block]

        results.extend(funcs)

    return results


def _trim_to_function(code: str) -> str:
    lines = code.split('\n')
    if not lines:
        return code
    def_indent = len(lines[0]) - len(lines[0].lstrip())
    last_body = 0
    for i in range(1, len(lines)):
        stripped = lines[i].lstrip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(stripped)
        if indent > def_indent:
            last_body = i
        else:
            break
    return '\n'.join(lines[:last_body + 1])

def rename_function(function: str, function_name: str) -> str:
    """
    Replace the name of the first function in `answer` with `function_name`.
    Only modifies the function name, keeps everything else intact.
    """
    pattern = r"def\s+(\w+)\s*\("

    new_answer = re.sub(pattern, f"def {function_name}(", function, count=1)
    return new_answer


def _exec_code_and_capture(code: str, conn: Connection, work_dir: Optional[str] = None):
    try:
        if work_dir is not None:
            os.makedirs(work_dir, exist_ok=True)
            os.chdir(work_dir)

        local_ns = {}
        exec(code, local_ns)

        for name, func in local_ns.items():
            if callable(func) and name.startswith("test_"):
                func()
        conn.send(True)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


class PyExecutor:

    def _run_with_timeout(self, code: str, timeout: int, work_dir: Optional[str] = "./code_stuff") -> Any:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=_exec_code_and_capture,
            args=(code, child_conn, work_dir)
        )
        
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError("Test execution timed out")

        if parent_conn.poll():
            result = parent_conn.recv()
            if isinstance(result, Exception):
                raise result
            return result
        else:
            raise RuntimeError("Child process terminated unexpectedly without sending a result.")

    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        success_tests = []
        failed_tests = []
        is_passing = True

        for test_code in tests:
            cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test_code, flags=re.MULTILINE)
            code_to_run = func + "\n" + cleaned_test
            try:
                self._run_with_timeout(code_to_run, timeout)
                success_tests.append(test_code)
            except Exception as e:
                failed_tests.append(f"{test_code}  # output: {e}")
                is_passing = False

        state = tuple(test in success_tests for test in tests)
        feedback = (
            "Tests passed:\n" + "\n".join(success_tests)
            + "\n\nTests failed:\n" + "\n".join(failed_tests)
        )
        return is_passing, feedback, state

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test, flags=re.MULTILINE)
        code_to_run = func + "\n" + cleaned_test
        try:
            self._run_with_timeout(code_to_run, timeout)
            return True
        except Exception:
            return False
        
    def check_code_report(self, completions: list[str], tests: list[str], test_info: list, timeout: int = 5) -> tuple[list[str], list[float]]:
        def extract_failed_tests(text: str) -> str:
            match = re.search(r"Tests failed:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else ""
        
        def extract_correct_function_name(text: str) -> str:
            match = re.search(r"from\s+solution\s+import\s+([a-zA-Z_]\w*)", text)
            return match.group(1) if match else ""

        reports = []
        avg_scores = [] 

        for completion, test_code_str, tf in zip(completions, tests, test_info):
            func_blocks = extract_python_code(completion.strip())
            collected_answer = '\n'.join(func_blocks)

            # correct_function_name = extract_correct_function_name(test_code_str)
            # if correct_function_name != "":
            #     collected_answer = rename_function(collected_answer, correct_function_name)
            collected_answer = rename_function(collected_answer, tf[0]["function_name"])

            test_block = extract_python_code(test_code_str.strip())
            test_list = [test_block[0] + "\n\n" + block for block in test_block[1:]]

            report_lines = []
            success_examples = 0

            for test in test_list:
                func_name_match = re.search(r"def\s+(test_\w+)\s*\(", test)
                func_name = func_name_match.group(1) if func_name_match else "unknown_test"

                is_passing, feedback, _ = self.execute(collected_answer, [test], timeout=timeout)

                if is_passing:
                    success_examples += 1
                    report_lines.append(f"✅ Test passed for '{func_name}'")
                else:
                    report_lines.append(f"❌ Test failed for '{func_name}': \n{extract_failed_tests(feedback)}")
            
            if len(test_list) != 0:
                avg_score = success_examples / len(test_list)
            else:
                avg_score = 1.0 
            avg_scores.append(avg_score)  
            report_lines.append(f"\nAverage correctness: {avg_score:.2f}")

            reports.append("\n".join(report_lines))

        return reports, avg_scores


if __name__ == "__main__":
    # ========== case 1: 两个函数中间夹了游离 assert ==========
    text1 = '''\
import math

def add(a, b):
    """Return sum."""
    return a + b

assert add(1, 2) == 3

def multiply(a, b):
    return a * b
'''
    funcs1 = extract_python_code(text1)
    print("=== case 1: 函数间夹 assert ===")
    for i, f in enumerate(funcs1):
        print(f"--- func {i} ---\n{f}\n")

    # ========== case 2: 多 return 语句 ==========
    text2 = '''\
def grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "C"
'''
    funcs2 = extract_python_code(text2)
    print("=== case 2: 多 return ===")
    for i, f in enumerate(funcs2):
        print(f"--- func {i} ---\n{f}\n")

    # ========== case 3: 纯 import 无函数 ==========
    text3 = "import math\nimport os\n"
    funcs3 = extract_python_code(text3)
    print("=== case 3: 纯 import ===")
    for i, f in enumerate(funcs3):
        print(f"--- func {i} ---\n{f}\n")

    # ========== case 4: 代码块中包含 markdown 标记 ==========
    text4 = '''\
```python
def foo(x):
    return x + 1
```
'''
    funcs4 = extract_python_code(text4)
    print("=== case 4: markdown 包裹 ===")
    for i, f in enumerate(funcs4):
        print(f"--- func {i} ---\n{f}\n")

    # ========== case 5: 函数后跟注释和空行 ==========
    text5 = '''\
def bar(x):
    y = x * 2
    return y

# this is a comment
'''
    funcs5 = extract_python_code(text5)
    print("=== case 5: 函数后跟注释 ===")
    for i, f in enumerate(funcs5):
        print(f"--- func {i} ---\n{f}\n")

        