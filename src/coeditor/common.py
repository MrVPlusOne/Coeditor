from dataclasses import dataclass
from typing import *

import libcst as cst
import textwrap
from textwrap import dedent
from pathlib import Path
from tqdm import tqdm
import subprocess
from spot.utils import assert_eq, show_string_diff, timed_action, TimeLogger
from IPython.display import display, HTML
import html

T1 = TypeVar("T1")
T2 = TypeVar("T2")

TokenSeq = list[int]


def run_command(args: Sequence[str], cwd: str | Path) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
    )


def splitlines(text: str) -> list[str]:
    """
    Split text into lines while preserving the newline characters at line ends.
    The last line will always be empty.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[-1]:
        lines.append("")
    return lines


def split_list(
    lst: list[T1],
    sep: T1,
    keep_sep: bool = False,
) -> list[list[T1]]:
    """
    Split a list into segments by a separator. When `keep_sep` is True, the
    separator will be kept in the resulting segments.
    The last segment will always be empty. This is a list version of `splitlines`.
    """

    result = list[list[T1]]()
    buff = list[T1]()
    for item in lst:
        if item == sep:
            if keep_sep:
                buff.append(sep)
            result.append(buff)
            buff = list[T1]()
        else:
            buff.append(item)
    result.append(buff)
    if not result or result[-1]:
        result.append([])
    return result


def join_list(
    segs: list[list[T1]],
    sep: T1 | None,
) -> list[T1]:
    result = list[T1]()
    for i, seg in enumerate(segs):
        if sep is not None and i > 0:
            result.append(sep)
        result.extend(seg)
    return result


HtmlCode = str


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
