import numpy as np
from coeditor.encoding import (
    AnalysisBasedEditEncoder,
    Del_id,
    FileBasedEditEncoder,
    CstBasedEditEncoder,
    TokenizedEdit,
    Newline_id,
    Add_id,
    WindowArgs,
    decode_tokens,
    encode_basic,
)
from spot.data import output_ids_as_seqs
from spot.utils import scalar_stats
from .common import *
from coeditor.history import (
    CommitInfo,
    ProjectEdit,
    get_commit_history,
    edits_from_commit_history,
)
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu


@dataclass
class TokenizedEditDataset:
    project2edits: dict[Path, list[TokenizedEdit]]

    def __repr__(self) -> str:
        n_projects = len(self.project2edits)
        n_edits = sum(len(edits) for edits in self.project2edits.values())
        return f"TokenizedEditDataset(n_projects={n_projects}, n_edits={n_edits})"

    def subset(self, repos: Iterable[Path]) -> "TokenizedEditDataset":
        return TokenizedEditDataset({repo: self.project2edits[repo] for repo in repos})

    def map(self, f: Callable[[TokenizedEdit], TokenizedEdit]):
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
        return {
            "n_projects": len(self.project2edits),
            "n_edits": len(input_sizes),
            "input_size": scalar_stats(input_sizes),
            "output_size": scalar_stats(output_sizes),
        }

    def all_edits(self) -> list[TokenizedEdit]:
        return join_list(self.project2edits.values())

    @staticmethod
    def from_edits(
        edits: Iterable[TokenizedEdit], path=Path("all")
    ) -> "TokenizedEditDataset":
        return TokenizedEditDataset({path: list(edits)})


EditEncoder = FileBasedEditEncoder | CstBasedEditEncoder | AnalysisBasedEditEncoder


def _process_commits(root: Path, commits: Sequence[CommitInfo], encoder: EditEncoder):
    edits = edits_from_commit_history(root, commits)
    tk_edits = list[TokenizedEdit]()
    if isinstance(encoder, AnalysisBasedEditEncoder):
        tk_edits.extend(encoder.encode_pedits(list(edits)))
    else:
        for pe in edits:
            tk_edits.extend(encoder.encode_pedit(pe))
    return tk_edits


def dataset_from_projects(
    project_roots: Sequence[Path],
    encoder: EditEncoder,
    max_history_per_repo: int = 1000,
    workers: int = DefaultWorkers,
) -> "TokenizedEditDataset":
    """
    Create a file-based editing dataset from the given list of project roots.
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
        history_chunk_size = max(50, math.ceil(len(h) / 4))
        for i in range(0, len(h), history_chunk_size):
            roots.append(root)
            # note that we need 1 extra overlapping commit to get all diffs
            chunked_histories.append(h[i : i + history_chunk_size + 1])
    tk_edits = pmap(
        _process_commits,
        roots,
        chunked_histories,
        [encoder] * len(roots),
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
    repos_root: Path,
    encoder: EditEncoder,
    max_history_per_repo: int = 1000,
    workers: int = DefaultWorkers,
) -> dict[str, TokenizedEditDataset]:
    splits = ["test", "valid", "train"]
    projects = dict[str, list[Path]]()
    for split in splits:
        if not (repos_root / split).exists():
            warnings.warn(f"Split {split} not found at {repos_root / split}.")
            continue
        ps = [p for p in (repos_root / split).iterdir() if p.is_dir]
        projects[split] = ps
        if not ps:
            warnings.warn(f"No projects found in {split} split")

    dataset = dataset_from_projects(
        join_list(projects.values()),
        encoder=encoder,
        max_history_per_repo=max_history_per_repo,
        workers=workers,
    )
    return {k: dataset.subset(v) for k, v in projects.items()}


import warnings

# turn off redundant BLEU warnings
warnings.simplefilter(
    "ignore",
    category=UserWarning,
    lineno=552,
)


def is_repetitive_edit(edit: TokenizedEdit, blue_threshold=0.8) -> bool:
    """Check if all additions in the output_tokens can be matched to
    an addition in the input_tokens with a BLEU score above the threshold."""

    def get_changes(tks, key_tk: Token):
        if tks and tks[0] == key_tk:
            s = decode_tokens(tks[1:])
            s.strip()
            return encode_basic(s)
        else:
            return []

    ctx_lines = split_list(edit.input_tks, Newline_id)
    main_lines = output_ids_as_seqs(edit.input_tks)
    ctx_addtions = [tks for l in ctx_lines if (tks := get_changes(l, Add_id))]
    ctx_deletions = [tks for l in ctx_lines if (tks := get_changes(l, Del_id))]

    def has_match(line, line_key: Token):
        if line:
            if line[0] == Add_id:
                added = line[1:]
                return any(
                    as_any(sentence_bleu([ref], added)) > blue_threshold
                    for ref in ctx_addtions
                )
            elif line == [Del_id]:
                deleted = main_lines[line_key]
                return any(
                    as_any(sentence_bleu([ref], deleted)) > blue_threshold
                    for ref in ctx_deletions
                )
            else:
                raise ValueError(f"Unexpected line: {decode_tokens(line)}")
        else:
            return True

    for k, seg in output_ids_as_seqs(edit.output_tks).items():
        for line in split_list(seg, Newline_id):
            if not has_match(line, k):
                return False
    return True


def save_datasets(datasets: dict[str, TokenizedEditDataset], save_dir: Path):
    for name, dataset in datasets.items():
        pickle_dump(save_dir / f"{name}.pkl", dataset)


def load_datasets(
    save_dir: Path, splits=("test", "valid", "train")
) -> dict[str, TokenizedEditDataset]:
    return {name: pickle_load(save_dir / f"{name}.pkl") for name in splits}
