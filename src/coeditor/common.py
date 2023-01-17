from argparse import ArgumentError
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import random
import sys
from typing import *

import libcst as cst
import textwrap
from textwrap import dedent
from pathlib import Path
from tqdm import tqdm
import subprocess
import enum
from spot.utils import (
    DefaultWorkers,
    assert_eq,
    groupby,
    show_expr,
    compute_line_diffs,
    show_string_diff,
    timed_action,
    TimeLogger,
    pmap,
    pfilter,
    pickle_load,
    pickle_dump,
    as_any,
    not_none,
    get_modified_args,
    repr_modified_args,
    PickleCache,
)
from IPython.display import display, HTML
import html
import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
import numbers
import ast
import warnings
import math
import itertools

T1 = TypeVar("T1")
T2 = TypeVar("T2")

Token = int
TokenSeq = list[Token]

RelPath = NewType("RelPath", Path)


def proj_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_dict() -> dict:
    if (path := proj_root() / "config" / "coeditor.json").exists():
        return json.loads(path.read_text())
    else:
        return {}


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
    return text.split("\n")


def count_lines(text: str) -> int:
    return len(text.split("\n"))


def split_list(
    lst: list[T1],
    sep: T1,
) -> list[list[T1]]:
    """
    Split a list into segments by a separator, always ends with an empty list.
    """
    result = list[list[T1]]()
    buff = list[T1]()
    for item in lst:
        if item == sep:
            result.append(buff)
            buff = list[T1]()
        else:
            buff.append(item)
    result.append(buff)
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


HtmlCode = str


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


@dataclass
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
            match node.body:
                case [ast.Expr(value=ast.Constant(value=str())), *body]:
                    node.body = body
            return node

        def visit_ClassDef(self, node: ast.ClassDef):
            node = cast(ast.ClassDef, self.generic_visit(node))
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
    except SyntaxError:
        return code


def code_equal(code1: str | cst.CSTNode, code2: str | cst.CSTNode) -> bool:
    if isinstance(code1, cst.CSTNode):
        code1 = show_expr(code1)
    if isinstance(code2, cst.CSTNode):
        code2 = show_expr(code2)
    if code1 == code2:
        return True
    code1 = normalize_code_by_ast(code1)
    code2 = normalize_code_by_ast(code2)
    return code1 == code2


@overload
def random_subset(
    all: Sequence[T1], n: int, rng: random.Random | None = None
) -> list[T1]:
    ...


@overload
def random_subset(
    all: Mapping[T1, T2], n: int, rng: random.Random | None = None
) -> dict[T1, T2]:
    ...


def random_subset(all, n: int, rng: random.Random | None = None):
    if rng is None:
        rng = random.Random(42)
    if isinstance(all, Sequence):
        xs = [x for x in all]
        rng.shuffle(xs)
        return xs[:n]
    elif isinstance(all, Mapping):
        keys = [k for k in all]
        rng.shuffle(keys)
        return {k: all[k] for k in keys[:n]}
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


def assert_str_match(actual: str, expect: str):
    if actual != expect:
        print_err("String difference:")
        diff = show_string_diff(expect, actual)
        print_err(diff)
        raise AssertionError("Strings didn't match.")
