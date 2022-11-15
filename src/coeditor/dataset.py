from coeditor.encoding import FileLevelEditTokenizer, TokenizedEdit
from .common import *
from coeditor.history import (
    CommitInfo,
    ProjectEdit,
    get_commit_history,
    edits_from_commit_history,
)


@dataclass
class TokenizedEditDataset:
    project2edits: dict[Path, list[TokenizedEdit]]

    def __repr__(self) -> str:
        n_projects = len(self.project2edits)
        n_edits = sum(len(edits) for edits in self.project2edits.values())
        return f"TokenizedEditDataset(n_projects={n_projects}, n_edits={n_edits})"


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
    history_chunk_size: int = 100,
    workers: int = DefaultWorkers,
) -> "TokenizedEditDataset":
    histories = [(p, get_commit_history(p)) for p in project_roots]
    roots = list[Path]()
    chunked_histories = list[list[CommitInfo]]()
    for root, h in histories:
        for i in range(0, len(h), history_chunk_size):
            roots.append(root)
            chunked_histories.append(h[i : i + history_chunk_size])
    tk_edits = pmap(
        _process_commits,
        roots,
        chunked_histories,
        desc="Created tokenized edits",
        max_workers=workers,
    )

    project2edits = dict[Path, list[TokenizedEdit]]()
    for root, edits in zip(roots, tk_edits):
        if root in project2edits:
            project2edits[root].extend(edits)
        else:
            project2edits[root] = edits

    return TokenizedEditDataset(project2edits)
