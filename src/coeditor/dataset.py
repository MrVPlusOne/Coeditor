import numpy as np
from coeditor.encoding import FileLevelEditTokenizer, TokenizedEdit
from .common import *
from coeditor.history import (
    CommitInfo,
    ProjectEdit,
    get_commit_history,
    edits_from_commit_history,
)
import pandas as pd


@dataclass
class TokenizedEditDataset:
    project2edits: dict[Path, list[TokenizedEdit]]

    def __repr__(self) -> str:
        n_projects = len(self.project2edits)
        n_edits = sum(len(edits) for edits in self.project2edits.values())
        return f"TokenizedEditDataset(n_projects={n_projects}, n_edits={n_edits})"

    def subset(self, repos: Iterable[Path]) -> "TokenizedEditDataset":
        return TokenizedEditDataset({repo: self.project2edits[repo] for repo in repos})

    def per_repo_stats(self) -> pd.DataFrame:
        rows = []
        for repo, edits in self.project2edits.items():
            avg_input_size = float(np.mean([len(e.input_tks) for e in edits]))
            avg_output_size = float(np.mean([len(e.output_tks) for e in edits]))
            rows.append(
                {
                    "repo": repo.name,
                    "n_edits": len(edits),
                    "avg_input_size": avg_input_size,
                    "avg_target_size": avg_output_size,
                }
            )
        return pd.DataFrame(rows)

    def all_edits(self) -> Iterable[TokenizedEdit]:
        for xs in self.project2edits.values():
            yield from xs


def _process_commits(root: Path, commits: Sequence[CommitInfo]):
    edits = edits_from_commit_history(root, commits)
    encoder = FileLevelEditTokenizer()
    tk_edits = list[TokenizedEdit]()
    for pe in edits:
        for me in pe.changes.values():
            tk_edits.extend(encoder.tokenize_edit(me))
    return tk_edits


def dataset_from_projects(
    project_roots: Sequence[Path],
    max_history_per_repo: int = 1000,
    history_chunk_size: int = 100,
    workers: int = DefaultWorkers,
) -> "TokenizedEditDataset":
    """
    Create a file-based editing datase from the given list of project roots.
    Args:
        - max_history_per_repo (int, optional): When the repo history is longer than
        this value, only the oldest portion is going to be used. Defaults to 1000.
        - history_chunk_size (int, optional): Would break the commit history into chunks
        of this size for parallel processing. Defaults to 100.
    """
    histories = pmap(
        get_commit_history,
        project_roots,
        max_workers=workers,
        desc="Getting commit histories",
        tqdm_args={"unit": "repo"},
    )
    # keep the oldest portion of the history
    histories = [commits[-max_history_per_repo:] for commits in histories]
    # break long commit sequences into chunks for parallelization
    roots = list[Path]()
    chunked_histories = list[list[CommitInfo]]()
    for root, h in zip(project_roots, histories):
        for i in range(0, len(h), history_chunk_size):
            roots.append(root)
            # note that we need 1 extra overlapping commit to get all diffs
            chunked_histories.append(h[i : i + history_chunk_size + 1])
    tk_edits = pmap(
        _process_commits,
        roots,
        chunked_histories,
        desc="Create tokenized edits",
        max_workers=workers,
        tqdm_args={"unit": "chunk"},
    )

    project2edits = dict[Path, list[TokenizedEdit]]()
    for root, edits in zip(roots, tk_edits):
        if root in project2edits:
            project2edits[root].extend(edits)
        else:
            project2edits[root] = edits

    return TokenizedEditDataset(project2edits)


def datasets_from_repos(
    repos_root: Path, max_history_per_repo: int = 1000, workers: int = DefaultWorkers
) -> dict[str, TokenizedEditDataset]:
    assert (repos_root / "train").is_dir()
    assert (repos_root / "test").is_dir()

    train_projects = [p for p in (repos_root / "train").iterdir() if p.is_dir()]
    test_projects = [p for p in (repos_root / "test").iterdir() if p.is_dir()]

    dataset = dataset_from_projects(
        train_projects + test_projects,
        max_history_per_repo=max_history_per_repo,
        workers=workers,
    )
    train_dataset = dataset.subset(train_projects)
    test_dataset = dataset.subset(test_projects)
    return {"train": train_dataset, "test": test_dataset}