import shutil
import tempfile
import traceback

from coeditor import scoped_changes
from coeditor._utils import pretty_print_dict, scalar_stats

from .c3problem import (
    C3Problem,
    C3ProblemGenerator,
    C3ProblemSimpleSplit,
    C3ProblemTokenizer,
    C3ProblemTransform,
    JediUsageAnalyzer,
    fix_jedi_cache,
)
from .change import Added
from .common import *
from .encoding import TEdit
from .git import CommitInfo, get_commit_history
from .scoped_changes import ProjectChangeProcessor, TProb, edits_from_commit_history


@dataclass
class TokenizedEditDataset(Generic[TEdit]):
    _edits: list[TEdit]

    def __repr__(self) -> str:
        n_edits = len(self.all_edits())
        return f"TokenizedEditDataset(n_edits={n_edits})"

    def subset_edits(self, n_edits: int) -> "TokenizedEditDataset":
        return TokenizedEditDataset.from_edits(self.all_edits()[:n_edits])

    def overall_stats(self) -> dict:
        all_edits = self.all_edits()
        n_added = sum(isinstance(e.change_type, Added) for e in all_edits)
        basic_stats = {
            "n_edits": len(all_edits),
            "n_additions": n_added,
        }
        extra_stats = dict[str, list]()
        for e in all_edits:
            for k, v in e.stats().items():
                if k in extra_stats:
                    extra_stats[k].append(v)
                else:
                    extra_stats[k] = [v]
        return basic_stats | {k: scalar_stats(v) for k, v in extra_stats.items()}

    def all_edits(self) -> list[TEdit]:
        return self._edits

    @staticmethod
    def from_edits(edits: Iterable[TEdit]) -> "TokenizedEditDataset[TEdit]":
        return TokenizedEditDataset(list(edits))


@dataclass
class C3CombinedEncoder:
    change_processor: ProjectChangeProcessor[C3Problem] = field(
        default_factory=C3ProblemGenerator
    )
    problem_tranform: C3ProblemTransform = field(default_factory=C3ProblemSimpleSplit)
    edit_tokenizer: C3ProblemTokenizer = field(default_factory=C3ProblemTokenizer)


@dataclass
class _ProcessingResult:
    edits: Sequence
    stats: dict[str, dict | Any]


def _process_commits(
    root: Path,
    workdir: Path,
    is_training: bool,
    max_history_per_repo: int,
    change_processor: ProjectChangeProcessor[C3Problem],
    cache: PickleCache,
    time_limit_per_commit: float = 10.0,
) -> _ProcessingResult:
    # use process-specific parso cache
    fix_jedi_cache(workdir)
    scoped_changes._tlogger.clear()
    change_processor.clear_stats()
    change_processor.set_training(is_training)
    key = f"{root.name}({max_history_per_repo}, {is_training=})"
    commits = []
    if not cache.contains(key):
        # keep the oldest commits
        commits = get_commit_history(root)[-max_history_per_repo:]
    try:
        # cannot return here since subprocess maybe be killed after returning
        edits = cache.cached(
            key,
            lambda: edits_from_commit_history(
                root,
                commits,
                tempdir=workdir / "code" / root.name,
                change_processor=change_processor,
                silent=True,
                time_limit=time_limit_per_commit * (len(commits) + 10),
            ),
        )
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        warnings.warn(f"Failed to process project: {root}\nError: {e}")
        traceback.print_exception(e, limit=-6)
        edits = []
    stats = dict()
    change_processor.append_stats(stats)
    rec_add_dict_to(stats, {"tlogger": scoped_changes._tlogger.times})
    return _ProcessingResult(edits, stats)


def dataset_from_projects(
    cache: PickleCache,
    project_roots: Sequence[Path],
    change_processor: ProjectChangeProcessor[TProb],
    repo_training: Sequence[bool],
    max_history_per_repo: int,
    time_limit_per_commit: float,
    workers: int = DefaultWorkers,
) -> "Mapping[Path, Sequence[TProb]]":
    """
    Create a TokenizedEditDataset from a list of project roots and a given encoder.
    Args:
        - max_history_per_repo (int, optional): When the repo history is longer than
        this value, only the oldest portion is going to be used. Defaults to 1000.
    """
    # get the process id
    pid = os.getpid()
    workdir = Path(tempfile.gettempdir()) / "dataset_from_projects" / f"pid-{pid}"

    roots = project_roots
    workdirs = [workdir / f"repo-{i}" for i in range(len(roots))]
    try:
        presults = pmap(
            _process_commits,
            roots,
            workdirs,
            repo_training,
            key_args={
                "max_history_per_repo": max_history_per_repo,
                "change_processor": change_processor,
                "time_limit_per_commit": time_limit_per_commit,
                "cache": cache,
            },
            max_workers=workers,
            tqdm_args={"unit": "repo"},
        )
    finally:
        if workdir.exists():
            print("Removing workdir:", workdir)
            shutil.rmtree(workdir)

    project2edits = dict[Path, list[TProb]]()

    try:
        stats = dict[str, Any]()
        for root, pr in zip(roots, presults):
            project2edits.setdefault(root, []).extend(pr.edits)
            rec_add_dict_to(stats, pr.stats)

        if "tlogger" in stats:
            df = TimeLogger.times_to_dataframe(stats.pop("tlogger"))
            print("Time stats:")
            display(df)
        if "analyzer_errors" in list(stats.keys()):
            errors: dict = stats.pop("analyzer_errors")
            for k in list(errors.keys()):
                if JediUsageAnalyzer.is_known_error(k):
                    errors.pop(k)
            if errors:
                print("Analyzer errors:")
                for k in sorted(errors.keys(), key=lambda k: errors[k], reverse=True):
                    print(f"{k}:\t{errors[k]}")
        if stats:
            print("Other Stats:")
            pretty_print_dict(stats)
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            print("Error while printing stats:", e)

    return project2edits


def datasets_from_repo_splits(
    cache: PickleCache,
    repos_root: Path,
    change_processor: ProjectChangeProcessor[TProb],
    splits: Sequence[str] = ("test", "valid", "train"),
    max_history_per_repo: int = 1000,
    time_limit_per_commit: float = 10.0,
    workers: int = DefaultWorkers,
) -> dict[str, Sequence[TProb]]:
    projects = dict[str, list[Path]]()
    split_is_training = dict[str, list[bool]]()
    for split in splits:
        if not (repos_root / split).exists():
            warnings.warn(f"Split {split} not found at {repos_root / split}.")
            continue
        ps = [p for p in (repos_root / split).iterdir() if p.is_dir]
        projects[split] = ps
        training = split == "train"
        split_is_training[split] = [training] * len(ps)
        if not ps:
            warnings.warn(f"No projects found in {split} split")

    dataset = dataset_from_projects(
        cache,
        join_list(projects.values()),
        change_processor=change_processor,
        repo_training=join_list(split_is_training.values()),
        time_limit_per_commit=time_limit_per_commit,
        max_history_per_repo=max_history_per_repo,
        workers=workers,
    )
    return {k: join_list(dataset[r] for r in repos) for k, repos in projects.items()}


class C3ProblemDataset(TypedDict, Generic[TProb]):
    train: Sequence[TProb]
    valid: Sequence[TProb]
    test: Sequence[TProb]


def make_or_load_dataset(
    dataset_name: str,
    change_processor: ProjectChangeProcessor[TProb],
    splits: Sequence[str],
    remake_problems: bool = False,
    time_limit_per_commit: float = 10.0,
    workers: int = DefaultWorkers,
) -> C3ProblemDataset[TProb]:
    prob_config = repr_modified_args(change_processor)
    processed_dir = get_dataset_dir(dataset_name) / "processed"
    cache_dir = processed_dir / prob_config
    cache = PickleCache(cache_dir)
    if remake_problems:
        cache.clear()
    results = datasets_from_repo_splits(
        cache,
        get_dataset_dir(dataset_name) / "repos",
        change_processor,
        workers=workers,
        splits=splits,
        time_limit_per_commit=time_limit_per_commit,
    )
    size_mb = 0.0
    n = 0
    for f in cache_dir.iterdir():
        n += 1
        size_mb += f.stat().st_size / (1024**2)
    print(f"Dataset total size ({n=}): {size_mb:.2f} MB")

    return C3ProblemDataset(
        train=results.get("train", []),
        valid=results.get("valid", []),
        test=results.get("test", []),
    )


def make_or_load_transformed_dataset(
    dataset_name: str,
    dataset: C3ProblemDataset | None,
    encoder: C3CombinedEncoder,
    remake_problems: bool = False,
    workers: int = DefaultWorkers,
) -> dict[str, Sequence[C3Problem]]:
    def transform_eval_problems(
        dataset: C3ProblemDataset,
    ) -> dict[str, Sequence[C3Problem]]:
        results = dict[str, Sequence[C3Problem]]()
        for split in ("valid", "test"):
            prob_lists = pmap(
                encoder.problem_tranform.transform,
                dataset[split],
                desc=f"transform({split})",
                chunksize=1000,
                max_workers=workers,
            )
            results[split] = join_list(prob_lists)
        return results

    proc_config = repr_modified_args(encoder.change_processor)
    trans_config = repr_modified_args(encoder.problem_tranform)
    transformed_dir = get_dataset_dir(dataset_name) / "transformed"
    cache = PickleCache(transformed_dir)
    return cache.cached(
        f"eval-{proc_config}-{trans_config}",
        lambda: transform_eval_problems(not_none(dataset)),
        remake=remake_problems,
    )


def save_datasets(datasets: Mapping[str, Any], save_dir: Path) -> None:
    for name, dataset in datasets.items():
        pickle_dump(save_dir / f"{name}.pkl", dataset)
    subprocess.run(["du", "-sh", save_dir])


def load_datasets(save_dir: Path, splits=("test", "valid", "train")) -> dict[str, Any]:
    return {
        name: pickle_load(path)
        for name in splits
        if (path := (save_dir / f"{name}.pkl")).exists()
    }


def get_repo_signature(repo: Path, n_commits: int = 30) -> tuple[str, ...]:
    # use the first n commits as the signature
    commits = get_commit_history(repo)[-n_commits:]
    return tuple(c.msg for c in commits)
