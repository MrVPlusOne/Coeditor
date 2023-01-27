from coeditor.common import *


@dataclass(frozen=True)
class CommitInfo:
    hash: str
    parents: tuple[str, ...]
    msg: str

    def summary(self) -> str:
        return f"[{self.hash[:10]} {short_str(self.msg)}]"


def get_commit_history(
    project_dir: Path,
    max_hisotry: int | None = None,
    commit_id: str = "HEAD",
) -> list[CommitInfo]:
    """Get the commit history of the project, start from the given `commit_id`,
    going backward in time.
    When a merge commit is encountered, the second parent (the branch that's
    being merged in) is used as the history.
    """
    commit_id = run_command(
        ["git", "rev-parse", commit_id],
        cwd=project_dir,
    ).strip()
    history = []
    for _ in range(max_hisotry if max_hisotry else 100000):
        lines = run_command(
            ["git", "cat-file", "-p", commit_id],
            cwd=project_dir,
        ).splitlines()
        parents = []
        for line in lines[1:]:
            if line.startswith("parent "):
                parents.append(line.split(" ")[1])
            else:
                break
        commit_msg = run_command(
            ["git", "show", commit_id, "-s", "--format=%s"],
            cwd=project_dir,
        ).strip()
        history.append(CommitInfo(commit_id, tuple(parents), commit_msg))
        if not parents:
            break
        commit_id = parents[-1]
    return history


def file_content_from_commit(
    project_dir: Path,
    commit: str,
    path: str,
) -> str:
    return run_command(
        ["git", "show", f"{commit}:{path}"],
        cwd=project_dir,
    )
