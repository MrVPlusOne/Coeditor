import numpy as np
from coeditor.encoding import (
    AnalysisBasedEditEncoder,
    Del_id,
    FileBasedEditEncoder,
    CstBasedEditEncoder,
    EditEncoder,
    TEdit,
    TokenizedEdit,
    Newline_id,
    Add_id,
    decode_tokens,
    encode_basic,
)
from spot.data import output_ids_as_seqs
from spot.utils import scalar_stats
from .common import *
from coeditor.history import (
    Added,
    CommitInfo,
    ProjectEdit,
    get_commit_history,
    edits_from_commit_history,
)
import pandas as pd


@dataclass
class TokenizedEditDataset(Generic[TEdit]):
    project2edits: dict[Path, list[TEdit]]

    def __repr__(self) -> str:
        n_projects = len(self.project2edits)
        n_edits = sum(len(edits) for edits in self.project2edits.values())
        return f"TokenizedEditDataset(n_projects={n_projects}, n_edits={n_edits})"

    def subset(self, repos: Iterable[Path]) -> "TokenizedEditDataset":
        return TokenizedEditDataset({repo: self.project2edits[repo] for repo in repos})

    def map(self, f: Callable[[TEdit], TEdit]) -> "TokenizedEditDataset[TEdit]":
        return TokenizedEditDataset(
            {repo: [f(e) for e in edits] for repo, edits in self.project2edits.items()}
        )

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

    def overall_stats(self) -> dict:
        input_sizes = [len(e.input_tks) for e in self.all_edits()]
        output_sizes = [len(e.output_tks) for e in self.all_edits()]
        n_added = sum(isinstance(e.change_type, Added) for e in self.all_edits())
        return {
            "n_projects": len(self.project2edits),
            "n_edits": len(input_sizes),
            "n_additions": n_added,
            "input_size": scalar_stats(input_sizes),
            "output_size": scalar_stats(output_sizes),
        }

    def all_edits(self) -> list[TEdit]:
        return join_list(self.project2edits.values())

    @staticmethod
    def from_edits(
        edits: Iterable[TEdit], path=Path("all")
    ) -> "TokenizedEditDataset[TEdit]":
        return TokenizedEditDataset({path: list(edits)})


def _process_commits(
    root: Path,
    commits: Sequence[CommitInfo],
    encoder: EditEncoder[TEdit],
    include_additions: bool,
) -> list[TEdit]:
    edits = edits_from_commit_history(root, commits)
    tk_edits = list()
    if isinstance(encoder, AnalysisBasedEditEncoder):
        tk_edits.extend(encoder.encode_pedits(list(edits), include_additions))
    else:
        for pe in edits:
            tk_edits.extend(encoder.encode_pedit(pe, include_additions))
    return tk_edits


def dataset_from_projects(
    project_roots: Sequence[Path],
    encoder: EditEncoder[TEdit],
    is_train: Sequence[bool],
    max_history_per_repo: int = 1000,
    workers: int = DefaultWorkers,
) -> "TokenizedEditDataset[TEdit]":
    """
    Create a TokenizedEditDataset from a list of project roots and a given encoder.
    Args:
        - max_history_per_repo (int, optional): When the repo history is longer than
        this value, only the oldest portion is going to be used. Defaults to 1000.
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
    include_additions = list[bool]()
    chunked_histories = list[list[CommitInfo]]()
    for root, h, train in zip(project_roots, histories, is_train):
        history_chunk_size = max(50, math.ceil(len(h) / 4))
        for i in range(0, len(h), history_chunk_size):
            roots.append(root)
            include_additions.append(train)
            # note that we need 1 extra overlapping commit to get all diffs
            chunked_histories.append(h[i : i + history_chunk_size + 1])
    tk_edits = pmap(
        _process_commits,
        roots,
        chunked_histories,
        [encoder] * len(roots),
        include_additions,
        desc="Create tokenized edits",
        max_workers=workers,
        tqdm_args={"unit": "chunk"},
    )  # return type cannot be inferred correctly without using TypeVarTuple
    project2edits = dict[Path, list[TEdit]]()
    for root, edits in zip(roots, tk_edits):
        edits = cast(list[TEdit], edits)
        if root in project2edits:
            project2edits[root].extend(edits)
        else:
            project2edits[root] = edits

    return TokenizedEditDataset(project2edits)


def datasets_from_repos(
    repos_root: Path,
    encoder: EditEncoder[TEdit],
    max_history_per_repo: int = 1000,
    predict_added_in_training: bool = True,
    workers: int = DefaultWorkers,
) -> dict[str, TokenizedEditDataset[TEdit]]:
    splits = ["test", "valid", "train"]
    projects = dict[str, list[Path]]()
    split_include_additions = dict[str, list[bool]]()
    for split in splits:
        if not (repos_root / split).exists():
            warnings.warn(f"Split {split} not found at {repos_root / split}.")
            continue
        ps = [p for p in (repos_root / split).iterdir() if p.is_dir]
        projects[split] = ps
        include_additions = predict_added_in_training and split == "train"
        split_include_additions[split] = [include_additions] * len(ps)
        if not ps:
            warnings.warn(f"No projects found in {split} split")

    dataset = dataset_from_projects(
        join_list(projects.values()),
        encoder=encoder,
        is_train=join_list(split_include_additions.values()),
        max_history_per_repo=max_history_per_repo,
        workers=workers,
    )
    return {k: dataset.subset(v) for k, v in projects.items()}


def save_datasets(datasets: dict[str, TokenizedEditDataset], save_dir: Path):
    for name, dataset in datasets.items():
        pickle_dump(save_dir / f"{name}.pkl", dataset)


def load_datasets(
    save_dir: Path, splits=("test", "valid", "train")
) -> dict[str, TokenizedEditDataset]:
    return {
        name: pickle_load(path)
        for name in splits
        if (path := (save_dir / f"{name}.pkl")).exists()
    }
