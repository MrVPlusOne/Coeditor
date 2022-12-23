from copy import deepcopy
import multiprocessing

from datasets import Dataset

from .data import (
    ChunkedDataset,
    CtxArgs,
    SrcCheckResult,
    SrcChunkInfo,
    TokenizedSrcSet,
    TokenizedSrc,
    code_to_check_from_preds,
    src_to_chunks_,
    feedbacks_to_tokenized_src,
)
from .model import DatasetPredResult, DecodingArgs, ModelWrapper, dynamic_dataloader
from .type_check import (
    MypyChecker,
    MypyFeedback,
    MypyResult,
    PythonType,
    normalize_type,
)
from .utils import *


class IncrSelector:
    "A strategy for selecting the best candidate time at each time step."
    pass


class SelectByOracle(IncrSelector):
    "Select the first candidate that matches the ground truth (if any)."
    pass


class SelectByCounting(IncrSelector):
    "Select the first candidate that has the least type errors."
    pass


def inline_single_prediction(
    src: TokenizedSrc, label_id: int, ty: PythonType, as_comment: bool
) -> "TokenizedSrc":
    tokenizer = DefaultTokenizer
    mask_id = tokenizer.mask_token_id
    to_insert = tokenizer.encode(str(ty), add_special_tokens=False)
    if as_comment:
        comment_start = tokenizer.encode("/* ", add_special_tokens=False)
        comment_end = tokenizer.encode(" */", add_special_tokens=False)
        to_insert = comment_start + to_insert + comment_end

    l_pos = src.types_pos[label_id]
    assert_eq(src.tokenized_code[l_pos], mask_id)

    new_code = src.tokenized_code[:l_pos] + to_insert + src.tokenized_code[l_pos + 1 :]
    # inlined_span = slice(l_pos, l_pos + len(to_insert))
    offset = len(to_insert) - 1
    new_types_pos = [
        pos + offset if i > label_id else pos for i, pos in enumerate(src.types_pos)
    ]

    return TokenizedSrc(
        file=src.file,
        repo=src.repo,
        types=src.types,
        types_pos=new_types_pos,
        types_str=src.types_str,
        types_tks=src.types_tks,
        types_info=src.types_info,
        main_code=src.main_code,
        tokenized_code=new_code,
        preamble_code=src.preamble_code,
        tokenized_preamble=src.tokenized_preamble,
        prev_types=None,  # don't need them for now
        inlined_spans=None,  # don't need them for now
        feedbacks=src.feedbacks,
    )


def sample_candidates(
    wrapper: ModelWrapper,
    src_data: TokenizedSrcSet,
    n_samples: int,
) -> tuple[ChunkedDataset, list[list[list[PythonType]]]]:
    ctx_args = wrapper.args.ctx_args

    do_sample = wrapper.args.do_sample
    if not do_sample:
        assert wrapper.args.num_beams is not None, "num_beams needs to be set"
        assert n_samples <= wrapper.args.num_beams

    chunks = src_data.to_chunks(ctx_args)
    n_chunks = len(chunks.data)

    if do_sample:
        samples = [
            wrapper.predict(chunks.data, tqdm_args={})
            for _ in tqdm(range(n_samples), desc="Sampling")
        ]  # of shape (n_samples, n_chunks, n_labels)
    else:
        samples = wrapper.predict(
            chunks.data,
            num_return_sequences=n_samples,
            tqdm_args={},
        )  # of shape (n_chunks, n_samples, n_labels)
        assert_eq(len(samples), n_chunks)
        assert_eq(len(samples[0]), n_samples)

    def get_preds(chunk_id, sample_id):
        return (
            samples[sample_id][chunk_id] if do_sample else samples[chunk_id][sample_id]
        )

    pred_candidates = [
        [get_preds(cid, sid) for sid in range(n_samples)] for cid in range(n_chunks)
    ]  # of shape (n_chunks, n_samples, n_labels)
    return chunks, pred_candidates


def select_candidates_by_type_errors(
    src_data: TokenizedSrcSet,
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
    only_same_file_error: bool = False,
) -> DatasetPredResult:
    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.all_srcs

    with src_data.setup_typechecking(srcs_to_check, skip_pre_fdbks=True) as env:
        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(len(chunks.data)):
            info = chunks.chunks_info[i]
            file = info.src_file
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = env.template_root / src.repo
            for j, candidates in enumerate(pred_candidates[i]):
                preds_dict = {
                    l_id: str(pred) for l_id, pred in zip(info.label_ids, candidates)
                }
                to_check[(i, j)] = (src, preds_dict, proj_root)

        to_check_values = to_check.values()
        check_rs: list[int] = pmap(
            count_type_errors_in_project,
            [[x[0]] for x in to_check_values],
            [[x[1]] for x in to_check_values],
            [x[2] for x in to_check_values],
            [only_same_file_error for _ in to_check_values],
            desc="map count_type_errors_in_project",
        )

    n_errors = dict(zip(to_check.keys(), check_rs))
    final_preds = list[list[PythonType]]()
    extra_info = list[dict]()
    for i in range(len(chunks.data)):
        candidates = pred_candidates[i]
        es = [n_errors[(i, j)] for j in range(len(candidates))]
        sample_id = int(np.argmin(es))
        final_preds.append(candidates[sample_id])
        extra_info.append({"n_errors": es[sample_id]})
    return DatasetPredResult(chunks, final_preds, extra_info)


def select_candidates_using_oracle(
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
) -> DatasetPredResult:
    final_preds = list[list[PythonType]]()
    extra_info = list[dict]()
    for i in tqdm(range(len(chunks.data)), desc="select_candidates_using_oracle"):
        info = chunks.chunks_info[i]
        candidates = pred_candidates[i]
        n_errors = []
        for preds in candidates:
            ne = sum(
                0 if normalize_type(p) == normalize_type(t) else 1
                for p, t in zip(preds, info.types)
            )
            n_errors.append(ne)
        sample_id = int(np.argmin(n_errors))
        final_preds.append(candidates[sample_id])
        extra_info.append({"n_errors_oracle": n_errors[sample_id]})

    return DatasetPredResult(chunks, final_preds, extra_info)


def select_first_candidates(
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
) -> DatasetPredResult[None]:
    final_preds = list[list[PythonType]]()
    for i in range(len(chunks.data)):
        preds = pred_candidates[i][0]
        final_preds.append(preds)

    return DatasetPredResult(chunks, final_preds, [])


@dataclass
class CriticAssesInfo:
    candidate_scores: list[float]
    candidate_label_scores: list[list[float]]

    @property
    def best_candidate(self) -> int:
        return int(np.argmax(self.candidate_scores))


def to_critic_inputs(
    src: TokenizedSrc,
    preds: dict[int, PythonType],
    check_r: SrcCheckResult,
    ctx_args: CtxArgs,
    labels_range: tuple[int, int] | None = None,
):
    """
    Patch each src with the type checker feedbacks and inline the previous predicitons,
    then break the src into one (if short enough) or more chunks.
    """
    errors, current_code = check_r
    fdbks = [] if isinstance(errors, str) else errors
    new_src = feedbacks_to_tokenized_src(
        src, current_code, fdbks, patch_predictions=False
    )
    new_src.prev_types = preds
    new_src = TokenizedSrc.inline_predictions(new_src, as_comment=False)
    chunks = list[dict]()
    chunks_info = list[SrcChunkInfo]()
    if labels_range is None:
        labels_range = min(preds.keys()), max(preds.keys()) + 1
    src_to_chunks_(chunks, chunks_info, new_src, labels_range, ctx_args)
    return chunks, chunks_info


def collect_type_errors_from_predictions(
    src_data: TokenizedSrcSet, result: DatasetPredResult, max_workers: int
) -> list[tuple[Path, MypyFeedback]]:
    "Apply all the predictioins and call the type checker once per project."

    chunks = result.chunks
    chunks_info = chunks.chunks_info
    chunk_preds = result.predictions

    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.all_srcs

    with src_data.setup_typechecking(srcs_to_check) as env:
        to_check = dict[Path, dict[Path, dict[int, str]]]()
        for i in range(len(chunks_info)):
            info = chunks.chunks_info[i]
            file = info.src_file
            src = file2src[file.relative_to(src_data.repos_root)]
            file = src.file
            proj_root = env.template_root / src.repo
            if proj_root not in to_check:
                to_check[proj_root] = dict()
            if file not in to_check[proj_root]:
                to_check[proj_root][file] = dict()
            pred_dict = to_check[proj_root][file]
            for l_id, pred in zip(info.label_ids, chunk_preds[i]):
                pred_dict[l_id] = str(pred)

        check_rs: list[MypyResult | str] = pmap(
            collect_type_errors_in_project,
            [[file2src[f] for f in d.keys()] for d in to_check.values()],
            [[preds for preds in d.values()] for d in to_check.values()],
            [root for root in to_check.keys()],
            max_workers=max_workers,
            desc="map collect_type_errors_in_project",
        )
        feebacks = [
            (f, e)
            for x in check_rs
            if isinstance(x, MypyResult)
            for f, ls in x.error_dict.items()
            for e in ls
        ]
    return feebacks


def count_type_errors_in_project(
    srcs: list[TokenizedSrc],
    preds_list: list[dict[int, str]],
    proj_root: Path,
    only_same_file_error: bool = False,
) -> int:
    r = collect_type_errors_in_project(srcs, preds_list, proj_root)
    if isinstance(r, MypyResult):
        if only_same_file_error:
            file_errors = [
                r.error_dict.get(s.file.relative_to(s.repo), []) for s in srcs
            ]
        else:
            file_errors = list(r.error_dict.values())
        return sum(len(ls) for ls in file_errors)
    else:
        return 0


def collect_type_errors_in_project(
    srcs: list[TokenizedSrc],
    preds_list: list[dict[int, str]],
    project_root: Path,
) -> MypyResult | str:
    # setup: copy all files into cwd
    proc = multiprocessing.current_process()
    cwd = (project_root.parent.parent / proc.name / project_root.name).resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    for f in project_root.glob("**/*.py"):
        rel_path = f.relative_to(project_root)
        (cwd / rel_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cwd / rel_path)

    for src, preds in zip(srcs, preds_list):
        rel_path = src.file.relative_to(src.repo)
        file_path = cwd / rel_path
        new_code = code_to_check_from_preds(src, preds)
        file_path.write_text(new_code)
    check_r = MypyChecker.check_project(cwd)
    if isinstance(check_r, MypyResult):
        check_r.error_dict = {
            f.relative_to(cwd): es for f, es in check_r.error_dict.items()
        }
    return check_r
