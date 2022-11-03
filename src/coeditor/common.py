from dataclasses import dataclass
from typing import *

import libcst as cst
from textwrap import dedent
from pathlib import Path
from tqdm import tqdm
import subprocess
from spot.utils import assert_eq, show_string_diff, timed_action, TimeLogger
from IPython.display import display

T1 = TypeVar("T1")
T2 = TypeVar("T2")

TokenSeq = list[int]


def run_command(args: Sequence[str], cwd: str | Path) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
    )


class Constants:
    TAB = " " * 4
