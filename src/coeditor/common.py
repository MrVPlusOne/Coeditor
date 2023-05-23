import ast
import asyncio
import enum
import html
import itertools
import json
import math
import numbers
import os
import random
import subprocess
import sys
import textwrap
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentError
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from textwrap import dedent
from typing import *

import numpy as np
from IPython.display import HTML, display
from tqdm import tqdm

from ._utils import (
    CodePosition,
    CodeRange,
    DefaultWorkers,
    PickleCache,
    TimeLogger,
    add_line_numbers,
    as_any,
    assert_all_eq,
    assert_eq,
    compute_line_diffs,
    compute_line_diffs_fast,
    cprint,
    get_modified_args,
    groupby,
    not_none,
    pfilter,
    pickle_dump,
    pickle_load,
    pmap,
    pretty_print_dict,
    repr_modified_args,
    run_long_task,
    scalar_stats,
    show_string_diff,
    split_dots,
    timed_action,
)

T1 = TypeVar("T1")
T2 = TypeVar("T2")

Token = int
TokenSeq = list[Token]

RelPath = NewType("RelPath", Path)
AbsPath = NewType("AbsPath", Path)


def to_rel_path(path: os.PathLike | str) -> RelPath:
    path = Path(path)
    if path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {path}")
    return RelPath(path)


def to_abs_path(path: Path) -> AbsPath:
    return AbsPath(path.resolve())


def proj_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_dict() -> dict:
    path = proj_root() / "config" / "coeditor.json"
    if not path.exists():
        warnings.warn(f"No config file found at `{path}`. Create a default one.")
        path.parent.mkdir(exist_ok=True)
        default_dict = {"datasets_root": "datasets_root", "models_root": "models_root"}
        path.write_text(json.dumps(default_dict, indent=4))
    return json.loads(path.read_text())


def get_config(key: str) -> str:
    d = get_config_dict()
    if key not in d:
        raise KeyError(f"Key '{key}' not found in `config/coeditor.json`")
    return d[key]


def get_gpu_id(default: int) -> int:
    if (s := os.getenv("GPU_ID")) is not None:
        return int(s)
    else:
        print("GPU_ID not set, using:", default)
        return default


def get_dataset_dir(dataname: str) -> Path:
    return Path(get_config("datasets_root")) / dataname


def get_model_dir(trained=True) -> Path:
    post = "trained" if trained else "training"
    return Path(get_config("models_root")) / post


def get_coeditor_model_path() -> str | Path:
    return "MrVPlusOne/coeditor-perm2k-base-v1.7.3"


def run_command(args: Sequence[str], cwd: str | Path) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
    )


async def start_command(args: Sequence[str], cwd: str | Path) -> str:
    assert args
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, cwd=cwd
    )
    stdout, stderror = await process.communicate()
    if stderror:
        raise RuntimeError(stderror.decode())
    return stdout.decode()


def splitlines(text: str) -> list[str]:
    """Split a text into lines and apalways ends with an empty line."""
    if not text:
        return []
    return text.split("\n")


def count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def fix_newline(text: str):
    if text.endswith("\n"):
        return text
    return text + "\n"


def split_list(
    lst: list[T1],
    sep: T1,
) -> list[list[T1]]:
    """
    Split a list into segments by a separator, always ends with an empty list.
    """
    if not lst:
        return []
    result = list[list[T1]]()
    ptr = 0
    for i, item in enumerate(lst):
        if item == sep:
            result.append(lst[ptr:i])
            ptr = i + 1
    result.append(lst[ptr:])
    return result


def join_list(
    segs: Iterable[Iterable[T1]],
    sep: T1 | None = None,
) -> list[T1]:
    result = list[T1]()
    for i, seg in enumerate(segs):
        if sep is not None and i > 0:
            result.append(sep)
        result.extend(seg)
    return result


TAB = " " * 4
SEP = "-" * 80
HtmlCode = str


def show_sections(
    *sections: tuple[str, str],
    sep: str = SEP,
) -> str:
    segs = list[str]()
    for title, content in sections:
        segs.append(sep)
        segs.append(f"{title}:")
        segs.append(content)
    return "\n".join(segs)


def print_sections(
    *sections: tuple[str, str],
    sep: str = SEP,
    file: TextIO = sys.stdout,
) -> None:
    print(show_sections(*sections, sep=sep), file=file)


def short_str(text: str, limit: int = 27) -> str:
    if len(text) <= limit:
        return text
    else:
        return text[:limit] + "..."


def display_html(html: HtmlCode, show_code=False) -> None:
    code = dedent(
        f"""\
        <html>
        <head>
            <link rel="stylesheet" href="https://the.missing.style/1.0.1/missing.min.css">
        </head>
        <body>
            {textwrap.indent(html, "    ")}
            <script type="module" src="https://the.missing.style/v1.0.1/missing-js/tabs.js"></script>
        </body>
        </html>
    """
    )
    if show_code:
        print(code)
    display(HTML(code))
    return None


def html_tabs(elems: list[Any], max_tabs: int = 16) -> HtmlCode:
    elems = elems[:max_tabs]
    buttons = "\n".join(
        [
            f"<button role='tab' aria-controls='tab-{i}' aria-selected={'true' if i==0 else 'false'}> {i} </button>"
            for i, _ in enumerate(elems)
        ]
    )

    contents = "\n".join(
        [
            f"<div id='tab-{i}' role='tabpanel' hidden> {html.escape(str(elem))} </div>"
            for i, elem in enumerate(elems)
        ]
    )

    code = dedent(
        f"""\
        <div role="tablist">
        {buttons}
        </div>
        {contents}
        """
    )
    return code


async def async_map(
    f: Callable[[T1], Awaitable[T2]], args: Iterable[T1] | AsyncIterable[T1]
) -> AsyncIterable[T2]:
    if isinstance(args, AsyncIterable):
        tasks = [f(x) async for x in args]
    else:
        tasks = [f(x) for x in args]
    for task in tasks:
        yield (await task)


class ExecutorHelper:
    """Create this inside a function to simplify `run_in_executor` calls."""

    def __init__(self, exec: Executor):
        self.eloop = asyncio.get_event_loop()
        self.exec = exec

    # TODO: specify more precise types using TypeVarTuple
    def run(self, f: Callable[..., T1], *args) -> asyncio.Future[T1]:
        return self.eloop.run_in_executor(self.exec, f, *args)

    async def map(self, f: Callable[..., T1], *args) -> list[T1]:
        tasks = [self.run(f, *arg) for arg in zip(*args)]
        return list(await asyncio.gather(*tasks))

    @contextmanager
    @staticmethod
    def cpu(workers: int = DefaultWorkers):
        with ProcessPoolExecutor(workers) as exec:
            yield ExecutorHelper(exec)


V = TypeVar("V")
W = TypeVar("W")


@dataclass(frozen=True)
class WeightedSum(Generic[V, W]):
    sum: V
    weight: W

    def average(self) -> float:
        if self.weight == 0:
            return float("nan")
        out = self.sum / self.weight  # type: ignore
        return float(out)

    def mean(self) -> float:
        return self.average()

    def __add__(self, other: "WeightedSum[V, W]") -> "WeightedSum[V, W]":
        sum = self.sum + other.sum  # type: ignore
        weight = self.weight + other.weight  # type: ignore
        return WeightedSum(sum, weight)

    def __repr__(self) -> str:
        return f"(mean={self.mean():.5g}, weight={self.weight})"


CountedSum = WeightedSum[int, int]


def normalize_code_by_ast(
    code: str, sort_keyargs: bool = True, remove_doc_string: bool = True
) -> str:
    """Normalize the code by parsing and unparsing it using the AST module.
    If parsing fails, return the original code."""

    class KeyargSorter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if node.keywords:
                node.keywords.sort(key=lambda x: x.arg or "None")
            return node

    class DocStringremover(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            return self._visit_def(node)

        def visit_Module(self, node: ast.Module) -> Any:
            return self._visit_def(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            return self._visit_def(node)

        def _visit_def(self, node):
            node = as_any(self.generic_visit(node))
            match node.body:
                case [ast.Expr(value=ast.Constant(value=str())), *body]:
                    node.body = body
            return node

    try:
        tree = ast.parse(dedent(code))
        if remove_doc_string:
            tree = DocStringremover().visit(tree)
        if sort_keyargs:
            tree = KeyargSorter().visit(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return code


def code_equal(code1: str, code2: str) -> bool:
    if code1 == code2:
        return True
    code1 = normalize_code_by_ast(code1)
    code2 = normalize_code_by_ast(code2)
    return code1 == code2


@overload
def random_subset(
    all: Sequence[T1], n: int, rng: random.Random | int | None = None
) -> list[T1]:
    ...


@overload
def random_subset(
    all: Mapping[T1, T2], n: int, rng: random.Random | int | None = None
) -> dict[T1, T2]:
    ...


def random_subset(all, n: int, rng: random.Random | int | None = None):
    if rng is None:
        rng = random.Random()
    elif isinstance(rng, int):
        rng = random.Random(rng)

    def _subset_ids(ids: list[int]):
        rng.shuffle(ids)
        ids = ids[:n]
        ids.sort()
        return ids

    if isinstance(all, Sequence):
        ids = list(range(len(all)))
        ids = _subset_ids(ids)
        xs = [all[i] for i in ids[:n]]
        return xs
    elif isinstance(all, Mapping):
        keys = [k for k in all]
        ids = list(range(len(keys)))
        ids = _subset_ids(ids)
        return {(k := keys[i]): all[k] for i in ids[:n]}
    else:
        raise ArgumentError(all, f"Unsupported arg type: {type(all)}")


def batched_map(
    xs: Sequence[T1],
    group_key: Callable[[T1], Any],
    f: Callable[[Sequence[T1]], Iterable[T2]],
) -> list[T2]:
    """
    Group the input elements `xs` into groups using the `group_key` function,
    run `f` on each group, then flatten the results into a single list while
    following the original order of `xs`.
    """
    groups = groupby(range(len(xs)), lambda i: group_key(xs[i]))
    batches = [[xs[i] for i in ids] for ids in groups.values()]
    results = [f(batch) for batch in batches]

    outputs = dict[int, T2]()
    for ids, result in zip(groups.values(), results):
        for i, y in zip(ids, result):
            outputs[i] = y
    return [outputs[i] for i in range(len(xs))]


TStamp = TypeVar("TStamp")


class TimedCache(Generic[T1, T2, TStamp]):
    """Store the time-stamped results to avoid recomputation."""

    def __init__(self) -> None:
        self.cache = dict[T1, tuple[TStamp, T2]]()

    def cached(self, key: T1, stamp: TStamp, f: Callable[[], T2]) -> T2:
        match self.cache.get(key):
            case (s, value) if stamp == s:
                return value
            case _:
                value = f()
                self.set(key, value, stamp)
                return value

    def set(self, key: T1, value: T2, stamp: TStamp) -> None:
        self.cache.pop(key, None)
        self.cache[key] = (stamp, value)


def print_err(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def assert_str_equal(actual: str, expect: str, name: str | None = None):
    actual = actual.rstrip()
    expect = expect.rstrip()
    if actual != expect:
        print_err(f"{expect = }")
        print_err(f"{actual = }")
        print_err("String difference:")
        diff = show_string_diff(expect, actual)
        print_err(diff)
        raise AssertionError(f"Strings didn't match: {name}")


def rec_add_dict_to(
    target: dict[str, Any],
    value: Mapping[str, Any],
    value_merger: Callable[[Any, Any], Any] = lambda x, y: x + y,
):
    for k, v in value.items():
        if isinstance(v, Mapping):
            if k not in target:
                target[k] = {}
            rec_add_dict_to(target[k], v, value_merger)
        elif isinstance(v, list):
            target.setdefault(k, []).extend(v)
        else:
            if k in target:
                target[k] = value_merger(target[k], v)
            else:
                target[k] = v


ElemPath = str
ModuleName = str


class ProjectPath(NamedTuple):
    """The path of a top-level function or method in a project."""

    module: ModuleName
    path: ElemPath

    def __str__(self) -> str:
        return f"{self.module}/{self.path}"

    def __repr__(self) -> str:
        return f"proj'{str(self)}'"

    def append(self, path: ElemPath) -> "ProjectPath":
        new_path = path if self.path == "" else f"{self.path}.{path}"
        return ProjectPath(self.module, new_path)

    def pop(self) -> "ProjectPath":
        p1 = ".".join(split_dots(self.path)[:-1])
        return ProjectPath(self.module, p1)

    @staticmethod
    def from_str(s: str) -> "ProjectPath":
        if "/" not in s:
            raise ValueError(f"A project path must have one '/': {s}")
        module, path = s.split("/")
        return ProjectPath(module, path)


def keystroke_cost(
    input: str,
    output: str,
    cursor_jump_cost: int = 4,
    init_curosr_dis: int | None = None,  # default to cursor_jump_cost
):
    """
    A string distance metric that takes the cost of moving the cursor into account.
    This metric aim to approximate the number of key strokes required to
    transform the input string into the output string.

    Starting with the state `i = len(input), j = len(output), cursor_dis = init_curosr_dis, deleting = False`,
    the cost is computed using the optimal combination of the following operations:
    - M: match char (cost=0), require `input[-i] == output[-j], not deleting`, cause
    `i -= 1, j -= 1, cursor_dis += 1`
    - D: delete input char (cost=1), require `cursor_dis == 0, not deleting`, cause`i -= 1`.
    - A: add output char (cost=1), require `cursor_dis == 0, not deleting`, cause`j -= 1`.
    - C: bring cursor here (cost=min(curosr_dis, cursor_jump_cost)), require nothing, cause`cursor_dis = 0`.
    - S: start deleting (cost=1), require `cursor_dis == 0, not deleting`, cause `deleting = True`.
    - K: keep deleting (cost=0), require `deleting`, cause`i -= 1`.
    - E: end deleting (cost=1), require `cursor_dis == 0, deleting`, cause`deleting = False`.

    Worst-case complexity: `len(input) * len(output) * cursor_jump_cost`.

    Unmodeled operations:
    - Copy and paste
    """
    l_in = len(input)
    l_out = len(output)
    MaxCost = l_in + l_out + cursor_jump_cost + 1000
    CacheKey = tuple[int, int, int, bool]
    costs = dict[CacheKey, int]()

    for c in range(cursor_jump_cost + 1):
        costs[(0, 0, c, False)] = 0
        costs[(0, 0, c, True)] = c + 1

    for i in range(l_in + 1):
        j_range = range(l_out + 1) if i != 0 else range(1, l_out + 1)
        i_char = input[-i] if i > 0 else None
        for j in j_range:
            j_char = output[-j] if j > 0 else None
            for cursor_dis in range(cursor_jump_cost + 1):
                # --- if deleting ---
                # 1: keep deleting
                new_dis = min(cursor_dis + 1, cursor_jump_cost)
                best_cost = costs[(i - 1, j, new_dis, True)] if i > 0 else MaxCost
                # 2: end deleting
                if cursor_dis > 0:
                    best_cost = min(best_cost, 1 + cursor_dis + costs[(i, j, 0, False)])
                costs[(i, j, cursor_dis, True)] = best_cost

                # --- if not deleting ---
                # 1: delete input char
                cost1 = costs[(i - 1, j, 0, False)] if i > 0 else MaxCost
                # 2: add output char
                cost2 = costs[(i, j - 1, 0, False)] if j > 0 else MaxCost
                # 3: start deleting
                cost3 = costs[(i, j, 0, True)]

                best_cost = min(cost1, cost2, cost3) + 1 + cursor_dis
                # match char
                if i_char == j_char:
                    new_dis = min(cursor_dis + 1, cursor_jump_cost)
                    best_cost = min(best_cost, costs[(i - 1, j - 1, new_dis, False)])
                costs[(i, j, cursor_dis, False)] = best_cost

    if init_curosr_dis is None:
        init_curosr_dis = cursor_jump_cost

    return costs[(l_in, l_out, init_curosr_dis, False)]


def keystroke_cost_rec(
    input: str,
    output: str,
    cursor_jump_cost: int = 4,
    init_curosr_dis: int | None = None,  # default to cursor_jump_cost
):
    """
    A string distance metric that takes the cost of moving the cursor into account.
    This metric aim to approximate the number of key strokes required to
    transform the input string into the output string.

    Starting with the state `i = len(input), j = len(output), cursor_dis = init_curosr_dis, deleting = False`,
    the cost is computed using the optimal combination of the following operations:
    - M: match char (cost=0), require `input[-i] == output[-j], not deleting`, cause
    `i -= 1, j -= 1, cursor_dis += 1`
    - D: delete input char (cost=1), require `cursor_dis == 0, not deleting`, cause`i -= 1`.
    - A: add output char (cost=1), require `cursor_dis == 0, not deleting`, cause`j -= 1`.
    - C: bring cursor here (cost=min(curosr_dis, cursor_jump_cost)), require nothing, cause`cursor_dis = 0`.
    - S: start deleting (cost=1), require `cursor_dis == 0, not deleting`, cause `deleting = True`.
    - K: keep deleting (cost=0), require `deleting`, cause`i -= 1`.
    - E: end deleting (cost=1), require `cursor_dis == 0, deleting`, cause`deleting = False`.

    Worst-case complexity: `len(input) * len(output) * cursor_jump_cost`.

    Unmodeled operations:
    - Copy and paste
    """
    l_in = len(input)
    l_out = len(output)
    MaxCost = l_in + l_out + cursor_jump_cost + 1000
    CacheKey = tuple[int, int, int, bool]
    cache = dict[CacheKey, int]()

    def rec(i: int, j: int, cursor_dis: int, deleting: bool) -> int:
        "Return the cost of matching input[i:] and output[j:]]."
        if i < 0 or j < 0:
            return MaxCost
        if i == 0 and j == 0 and not deleting:
            return 0  # don't need to care about curosr in this case

        key = (i, j, cursor_dis, deleting)
        if key in cache:
            return cache[key]

        if deleting:
            # keep deleting
            new_dis = min(cursor_dis + 1, cursor_jump_cost)
            best_cost = rec(i - 1, j, new_dis, deleting=True)
            # end deleting
            if cursor_dis > 0:
                best_cost = min(
                    best_cost, 1 + cursor_dis + rec(i, j, cursor_dis=0, deleting=False)
                )
        else:
            # delete input char
            cost1 = rec(i - 1, j, 0, False)
            # add output char
            cost2 = rec(i, j - 1, 0, False)
            # start deleting
            cost3 = rec(i, j, 0, True)

            best_cost = min(cost1, cost2, cost3) + 1 + cursor_dis
            # match char
            if i > 0 and j > 0 and input[-i] == output[-j]:
                new_dis = min(cursor_dis + 1, cursor_jump_cost)
                best_cost = min(best_cost, rec(i - 1, j - 1, new_dis, False))
        cache[key] = best_cost
        return best_cost

    if init_curosr_dis is None:
        init_curosr_dis = cursor_jump_cost

    return rec(l_in, l_out, init_curosr_dis, False)


def keystroke_cost_old(
    input: str,
    output: str,
    cursor_jump_cost: int = 4,
    init_curosr_dis: int | None = None,  # default to cursor_jump_cost
):
    """
    A string distance metric that takes the cost of moving the cursor into account.
    This metric aim to approximate the number of key strokes required to
    transform the input string into the output string.

    Starting with the state `i = 0, j = 0, cursor_dis = init_curosr_dis, deleting = False`,
    the cost is computed using the optimal combination of the following operations:
    - M: match char (cost=0), require `input[i] == output[j], not deleting`, cause
    `i += 1, j += 1, cursor_dis += 1`
    - D: delete input char (cost=1), require `cursor_dis == 0, not deleting`, cause`i += 1`.
    - A: add output char (cost=1), require `cursor_dis == 0, not deleting`, cause`j += 1`.
    - C: bring cursor here (cost=min(curosr_dis, cursor_jump_cost)), require nothing, cause`cursor_dis = 0`.
    - S: start deleting (cost=1), require `cursor_dis == 0, not deleting`, cause `deleting = True`.
    - K: keep deleting (cost=0), require `deleting`, cause`i += 1`.
    - E: end deleting (cost=1), require `cursor_dis == 0, deleting`, cause`deleting = False`.

    Worst-case complexity: `len(input) * len(output) * cursor_jump_cost`.

    Unmodeled operations:
    - Copy and paste
    """
    l_in = len(input)
    l_out = len(output)
    MaxCost = l_in + l_out + cursor_jump_cost + 1000
    CacheKey = tuple[int, int, int, bool]
    cache = dict[CacheKey, int]()

    def rec(i: int, j: int, cursor_dis: int, deleting: bool) -> int:
        "Return the cost of matching input[i:] and output[j:]]."
        if i > l_in or j > l_out:
            return MaxCost
        if i == l_in:
            if j == l_out and not deleting:
                return 0  # don't need to care about curosr in this case
            # type out all remaining chars
            return cursor_dis + int(deleting) + (l_out - j)

        key = (i, j, cursor_dis, deleting)
        if key in cache:
            return cache[key]

        if deleting:
            # end deleting
            if cursor_dis > 0:
                cost0 = 1 + cursor_dis + rec(i, j, cursor_dis=0, deleting=False)
            else:
                cost0 = MaxCost  # not an option
            # keep deleting
            new_dis = min(cursor_dis + 1, cursor_jump_cost)
            cost1 = rec(i + 1, j, new_dis, deleting=True)

            best_cost = min(cost0, cost1)
        else:
            # match char
            if i < l_in and j < l_out and input[i] == output[j]:
                new_dis = min(cursor_dis + 1, cursor_jump_cost)
                cost0 = rec(i + 1, j + 1, new_dis, False)
            else:
                cost0 = MaxCost  # not an option
            # delete input char
            cost1 = 1 + rec(i + 1, j, 0, False) + cursor_dis
            # add output char
            cost2 = 1 + rec(i, j + 1, 0, False) + cursor_dis
            # start deleting
            cost3 = 1 + rec(i, j, 0, True) + cursor_dis

            best_cost = min(cost0, cost1, cost2, cost3)
        cache[key] = best_cost
        return best_cost

    if init_curosr_dis is None:
        init_curosr_dis = cursor_jump_cost

    return rec(0, 0, init_curosr_dis, False)


def bootstrap_sample(
    values: Sequence[float], iterations: int = 10000
) -> tuple[float, float]:
    """Return the bootstrap estimate of the confidence interval of the mean."""
    n = len(values)
    means = list[float]()
    for _ in range(iterations):
        mu = float(np.mean(np.random.choice(values, size=n, replace=True)))
        means.append(mu)
    means.sort()
    return means[int(iterations * 0.025)], means[int(iterations * 0.975)]


def bootstrap_compare(
    this_perf: Sequence[float | bool],
    that_perf: Sequence[float | bool],
    iterations: int = 10000,
) -> float:
    """Return the p-value of "this is better than that" using bootstrap sampling."""
    n = len(this_perf)
    assert_eq(len(that_perf), n)
    this_array = np.array(this_perf)
    that_array = np.array(that_perf)

    outcomes = list[bool]()
    for _ in range(iterations):
        ids = np.random.choice(range(n), size=n, replace=True)
        outcome = bool(np.mean(this_array[ids]) >= np.mean(that_array[ids]))
        outcomes.append(outcome)
    return float(np.mean(outcomes))
