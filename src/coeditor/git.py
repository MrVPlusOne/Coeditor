import shutil
from datetime import datetime

import dateparser

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
    max_history: int | None = None,
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
    for _ in range(max_history if max_history else 100000):
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


@dataclass
class GitRepo:
    author: str
    name: str
    url: str
    stars: int
    forks: int
    description: str
    license: str
    archived: bool
    last_update: Optional[datetime] = None
    num_commits: Optional[int] = None

    def authorname(self):
        return f"{self.author}~{self.name}"

    def get_root(self, repos_dir: Path) -> Path:
        return repos_dir / "downloaded" / self.authorname()

    def download(
        self, repos_dir: Path, full_history: bool = True, timeout=None
    ) -> bool:
        depth = "--depth=1" if not full_history else ""
        subprocess.run(
            ["git", "clone", *depth, self.url, self.authorname()],
            cwd=(repos_dir / "downloading"),
            timeout=timeout,
            capture_output=True,
        )
        if not (repos_dir / "downloading" / self.authorname()).is_dir():
            # git clone failed. Possibly caused by invalid url?
            return False
        shutil.move(
            repos_dir / "downloading" / self.authorname(), (repos_dir / "downloaded")
        )
        return True

    def read_last_update(self, repos_dir):
        d = self.get_root(repos_dir)
        s = subprocess.run(
            ["git", "log", "-1", "--format=%cd"], cwd=d, capture_output=True, text=True
        ).stdout
        lu = dateparser.parse(s.split("+")[0])
        assert lu is not None
        self.last_update = lu.replace(tzinfo=None)
        return self.last_update

    def count_lines_of_code(self, repos_dir):
        n_lines = 0
        for src in self.get_root(repos_dir).glob("**/*.py"):
            with open(src, "r") as fp:
                n_lines += sum(1 for line in fp if line.rstrip())
        self.lines_of_code = n_lines
        return n_lines

    def count_commits(self, repos_dir) -> int:
        result = run_command(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=self.get_root(repos_dir),
        )
        n = int(result)
        self.num_commits = n
        return n

    def revert_changes(self, repos_dir):
        rd = self.get_root(repos_dir)
        result = subprocess.run(
            ["git", "diff", "--name-only"], cwd=rd, capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip() != "":
            print("Reverting changes in", rd)
            subprocess.run(
                ["git", "checkout", "."],
                cwd=rd,
            )

    @staticmethod
    def from_github_item(item: dict):
        return GitRepo(
            author=item["owner"]["login"],
            name=item["name"],
            url=item["html_url"],
            description=item["description"],
            license=item["license"]["key"],
            stars=item["stargazers_count"],
            forks=item["forks_count"],
            archived=item["archived"],
            last_update=not_none(dateparser.parse(item["pushed_at"])).replace(
                tzinfo=None
            ),
        )
