# -----------------------------------------------------------
# project-level static analysis

import copy
from functools import cache, lru_cache, cached_property
import enum

from libcst import MetadataWrapper

from spot import PythonType
from .type_check import parse_type_expr
from .type_env import AccuracyMetric, AnnotCat, AnnotPath, type_accuracies

from .utils import *
from libcst.metadata import (
    CodeRange,
    QualifiedNameProvider,
    QualifiedName,
    QualifiedNameSource,
    PositionProvider,
)
from libcst._metadata_dependent import LazyValue
from pyrsistent import pmap as persist_map, PMap

ModuleName = str
ElemPath = str

CNode = TypeVar("CNode", bound=cst.CSTNode)


def remove_imports(
    m: cst.Module,
) -> tuple[cst.Module, list[cst.Import | cst.ImportFrom]]:
    """Removes all top-level import statements and collect them into a list."""
    remover = ImportsRemover()
    m = m.visit(remover)
    return m, list(remover.import_stmts)


def remove_comments(m: CNode) -> CNode:
    """Removes all comments and docstrings."""
    return cast(CNode, m.visit(CommentRemover()))


def remove_empty_lines(m: CNode) -> CNode:
    m = cast(CNode, m.visit(EmptyLineRemove()))
    return m


def remove_types(m: CNode, type_mask="...") -> CNode:
    """Removes all type annotations when possible; otherwise replace
    with the provided type_mask"""
    return cast(CNode, m.visit(AnnotRemover(type_mask)))


def mask_types(m: CNode, type_mask=SpecialNames.TypeMask) -> CNode:
    """Replace all type annotations with the provided type_mask"""

    class AnnotMasker(cst.CSTTransformer):
        def leave_Annotation(self, node, updated: cst.Annotation):
            return updated.with_changes(annotation=cst.Name(value=type_mask))

        def visit_Param(self, node: "cst.Param") -> Optional[bool]:
            # skip the types on the `self` parameter.
            return node.name.value != "self"

    return cast(CNode, m.visit(AnnotMasker()))


class ProjectPath(NamedTuple):
    """The path of a top-level function or method in a project."""

    module: ModuleName
    path: ElemPath

    def __str__(self) -> str:
        return f"{self.module}/{self.path}"

    def __repr__(self) -> str:
        return f"proj'{str(self)}'"

    def append(self, path: ElemPath) -> "ProjectPath":
        new_path = path if self.path == "" else f"{self.path}.{path}"
        return ProjectPath(self.module, new_path)

    def pop(self) -> "ProjectPath":
        p1 = ".".join(self.path.split(".")[:-1])
        return ProjectPath(self.module, p1)

    @staticmethod
    def from_str(s: str) -> "ProjectPath":
        if "/" not in s:
            raise ValueError(f"A project path must have one '/': {s}")
        module, path = s.split("/")
        return ProjectPath(module, path)

    @staticmethod
    def annot_path_to_module_path(p: AnnotPath) -> ElemPath:
        def simplify(seg: str):
            if seg.endswith("]"):
                i = seg.find("[")
                return seg[:i]
            else:
                return seg

        segs = [simplify(s) for s in reversed(p.value)]
        match segs:
            case [*parent, m, v] if v.startswith("self."):
                segs = [*parent, v[5:]]
        return ".".join(segs)

    @staticmethod
    def from_annot_path(rel_path: Path, p: AnnotPath) -> "ProjectPath":
        mname = PythonProject.rel_path_to_module_name(rel_path)
        mpath = ProjectPath.annot_path_to_module_path(p)
        return ProjectPath(mname, mpath)


ProjNamespace = type(persist_map(dict[str, ProjectPath]()))


def get_decorator_name(dec: cst.Decorator) -> str | None:
    match dec.decorator:
        case cst.Name(n):
            return n
        case cst.Attribute(attr=cst.Name(n)):
            return n
        case cst.Call(func=cst.Name(n) | cst.Attribute(attr=cst.Name(n))):
            return n
        case _:
            return None


@dataclass
class LabelInfo:
    annot: cst.Annotation
    cat: AnnotCat
    attached_to: cst.CSTNode


@dataclass
class PythonFunction:
    name: str
    path: ProjectPath
    parent_class: ProjectPath | None
    tree: cst.FunctionDef

    def __repr__(self):
        if self.in_class:
            return f"ClassMethod(path={self.path})"
        else:
            return f"GlobalFunction(path={self.path})"

    @cached_property
    def code(self) -> str:
        return show_element(self.tree, self.in_class)

    @cached_property
    def header_lines(self) -> int:
        headerless = self.tree.with_changes(body=cst.IndentedBlock([]))
        return len(show_expr(headerless).split("\n")) - 1

    @cached_property
    def header_body_code(self) -> tuple[str, str]:
        lines = self.code.split("\n")
        header = "\n".join(lines[: self.header_lines])
        body = "\n".join(lines[self.header_lines :])
        return header, body

    @property
    def in_class(self) -> bool:
        return self.parent_class is not None

    def get_signature(self) -> "FunctionSignature":
        return FunctionSignature.from_function(self.tree, self.parent_class is not None)

    def get_labels(self) -> list[LabelInfo]:
        is_method = self.in_class
        labels = list[LabelInfo]()

        class LabelsFinder(cst.CSTVisitor):
            def visit_Param(self, node: "cst.Param") -> Optional[bool]:
                if not (is_method and node.name.value == "self") and node.annotation:
                    label = LabelInfo(node.annotation, AnnotCat.FuncArg, node.name)
                    labels.append(label)

        self.tree.params.visit(LabelsFinder())
        rannot = self.tree.returns
        if rannot is not None:
            return_info = LabelInfo(rannot, AnnotCat.FuncReturn, self.tree)
            labels.append(return_info)
        return labels

    @cached_property
    def is_fixture(self) -> bool:
        for dec in self.tree.decorators:
            if get_decorator_name(dec) == "fixture":
                return True
        return False

    @cached_property
    def is_test_func(self) -> bool:
        # follow the pytest rules (but ignore the method requirements):
        # https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#conventions-for-python-test-discovery
        return self.name.startswith("test") and (
            (file := self.path.module.split(".")[-1]).startswith("test_")
            or file.endswith("_test")
        )

    @cached_property
    def is_fixture_user(self) -> bool:
        # both tests and fixtures can use other fixtures
        return self.is_test_func or self.is_fixture

    def remove_comments(self) -> Self:
        new_tree = remove_comments(self.tree)
        return PythonFunction(self.name, self.path, self.parent_class, new_tree)


@dataclass
class PythonVariable:
    name: str
    path: ProjectPath
    parent_class: ProjectPath | None
    tree: cst.Name | cst.Attribute
    assignments: list[
        cst.Assign | cst.AnnAssign
    ]  # only record assignments outside of functions
    wrapped_assignments: list[
        cst.Assign | cst.AnnAssign
    ]  # assignments inside functions

    def __repr__(self):
        if self.in_class:
            return f"ClassAttribute(path={self.path})"
        else:
            return f"GlobalVariable(path={self.path})"

    @cached_property
    def code(self) -> str:
        if self.assignments:
            return "\n".join(show_element(a, self.in_class) for a in self.assignments)
        else:
            stmt = cst.AnnAssign(cst.Name(self.name), cst.Annotation(cst.Ellipsis()))
            return show_element(stmt, self.in_class)

    @property
    def in_class(self) -> bool:
        return self.parent_class is not None

    def get_signature(self) -> "VariableSignature":
        annot = None
        for a in self.assignments:
            if isinstance(a, cst.AnnAssign):
                annot = a.annotation
                break  # the first annotation wins
        return VariableSignature(annot, self.parent_class is not None)

    def get_labels(self) -> Iterable[LabelInfo]:
        sig = self.get_signature()
        if sig.annot is not None:
            cat = AnnotCat.ClassAtribute if self.in_class else AnnotCat.GlobalVar
            yield LabelInfo(sig.annot, cat, self.tree)

    def iter_rhs(self) -> Iterable[cst.BaseExpression]:
        for a in self.assignments + self.wrapped_assignments:
            if isinstance(a, cst.AnnAssign) and a.value:
                yield a.value
            elif isinstance(a, cst.Assign) and a.value:
                yield a.value

    def remove_comments(self) -> Self:
        new_assignments = [remove_comments(a) for a in self.assignments]
        new_wrapped_assignments = [remove_comments(a) for a in self.wrapped_assignments]
        return PythonVariable(
            self.name,
            self.path,
            self.parent_class,
            self.tree,
            new_assignments,
            new_wrapped_assignments,
        )


PythonElem = PythonFunction | PythonVariable


def show_element(tree: cst.CSTNode, indent: bool) -> str:
    code = show_expr(tree).strip("\n")
    if indent:
        code = textwrap.indent(code, "    ")
    return code


@dataclass
class VariableSignature:
    annot: cst.Annotation | None
    in_class: bool

    def __repr__(self):
        content = "MISSING" if self.annot is None else show_expr(self.annot.annotation)
        if self.in_class:
            return f"AttrSig({content})"
        else:
            return f"VarSig({content})"

    def n_annotated(self) -> int:
        return int(self.annot is not None)

    def n_annots(self) -> int:
        return 1

    def get_annotated(self):
        if self.annot is not None:
            cat = AnnotCat.ClassAtribute if self.in_class else AnnotCat.GlobalVar
            yield cat, self.annot

    def updated(self, other: "VariableSignature") -> "VariableSignature":
        if other.annot is not None:
            return VariableSignature(other.annot, self.in_class)
        else:
            return self

    def drop_types(self) -> "VariableSignature":
        return VariableSignature(None, self.in_class)


@dataclass
class FunctionSignature:
    params: dict[str, cst.Annotation | None]
    returns: cst.Annotation | None
    in_class: bool

    def __str__(self):
        param_strs = [
            p if a is None else f"{p}: {show_expr(a.annotation, False)}"
            for p, a in self.params.items()
        ]
        return_str = (
            "MISSING"
            if self.returns is None
            else show_expr(self.returns.annotation, False)
        )
        return f"({', '.join(param_strs)}) -> {return_str}"

    def __repr__(self):
        head = "MethodSig" if self.in_class else "FuncSig"
        return f"{head}({str(self)})"

    def n_annotated(self) -> int:
        return sum(a is not None for a in self.params.values()) + (
            self.returns is not None
        )

    def n_annots(self) -> int:
        return len(self.params) + 1

    def get_annotated(self) -> Generator[tuple[AnnotCat, cst.Annotation], None, None]:
        for a in self.params.values():
            if a is not None:
                yield AnnotCat.FuncArg, a
        if self.returns is not None:
            yield AnnotCat.FuncReturn, self.returns

    def updated(self, other: "FunctionSignature") -> "FunctionSignature":
        params = copy.deepcopy(self.params)
        for p, a in other.params.items():
            if a is not None:
                params[p] = a
        returns = other.returns if other.returns is not None else self.returns
        return FunctionSignature(params, returns, self.in_class)

    def drop_types(self) -> "FunctionSignature":
        return FunctionSignature({p: None for p in self.params}, None, self.in_class)

    @staticmethod
    def from_function(func: cst.FunctionDef, in_class: bool) -> "FunctionSignature":
        extractor = FunctionSignature._ParamsExtractor()
        func.params.visit(extractor)
        return FunctionSignature(extractor.annots, func.returns, in_class)

    def apply(self, func: cst.FunctionDef) -> cst.FunctionDef:
        applier = FunctionSignature._ParamsApplier(self)
        new_params = func.params.visit(applier)
        assert isinstance(new_params, cst.Parameters)
        return func.with_changes(params=new_params, returns=self.returns)

    class _ParamsExtractor(cst.CSTVisitor):
        def __init__(self):
            self.annots = dict[str, cst.Annotation | None]()

        def visit_Param(self, node: "cst.Param") -> Optional[bool]:
            if node.name.value != "self":
                self.annots[node.name.value] = node.annotation
            return False  # skip default expressions

    class _ParamsApplier(cst.CSTTransformer):
        def __init__(self, func_signature: "FunctionSignature"):
            self.params = func_signature.params

        def visit_Param(self, node):
            return False  # skip default expressions

        def leave_Param(self, node, updated: "cst.Param"):
            if updated.name.value == "self":
                return updated
            pname = updated.name.value
            if pname not in self.params:
                raise LookupError(f"param {pname} not in {self.params.keys()}")
            return updated.with_changes(annotation=self.params[pname])


ElemSignature = VariableSignature | FunctionSignature
SignatureMap = dict[ProjectPath, ElemSignature]


def reorder_signature_map(
    pred_map: SignatureMap, label_map: SignatureMap
) -> SignatureMap:
    """Reorder the function args in the prediction to match the labels (if present)."""
    new_sigmap = SignatureMap()
    for path in label_map:
        match pred_map.get(path), label_map.get(path):
            case FunctionSignature(params=pred_params) as pred_sig, FunctionSignature(
                params=label_params
            ):
                new_params = {
                    p: pred_params[p] for p in label_params if p in pred_params
                }
                for p in pred_params:
                    if p not in new_params:
                        new_params[p] = pred_params[p]
                new_sigmap[path] = FunctionSignature(
                    new_params, pred_sig.returns, pred_sig.in_class
                )
            case _:
                pass

    for path in pred_map:
        if path not in new_sigmap:
            new_sigmap[path] = pred_map[path]

    return new_sigmap


from functools import cached_property


@dataclass
class PythonClass:
    name: str
    path: ProjectPath
    attributes: dict[str, PythonVariable]
    methods: dict[str, PythonFunction]
    inner_classes: dict[str, "PythonClass"]
    tree: cst.ClassDef
    parent_class: ProjectPath | None

    def __repr__(self):
        return f"PythonClass(path={self.path}, n_attrs={len(self.attributes)}, n_methods={len(self.methods)})"

    def all_elements(self) -> Generator[PythonElem, None, None]:
        yield from self.attributes.values()
        yield from self.methods.values()
        for c in self.inner_classes.values():
            yield from c.all_elements()


@dataclass
class PythonModule:
    functions: list[PythonFunction]
    global_vars: list[PythonVariable]
    classes: list[PythonClass]
    name: ModuleName
    # an over-approximation of the set of imported modules
    imported_modules: set[ModuleName]
    defined_symbols: dict[str, ProjectPath]
    tree: cst.Module
    location_map: dict[cst.CSTNode, CodeRange]
    elem2pos: dict[ElemPath, CodeRange]
    removed_comments: list[cst.CSTNode]

    @cached_property
    def classes_dict(self) -> dict[ElemPath, PythonClass]:
        return {c.path.path: c for c in self.all_classes()}

    @cached_property
    def elems_dict(self) -> dict[ElemPath, PythonElem]:
        return {e.path.path: e for e in self.all_elements()}

    @cached_property
    def code(self) -> str:
        return self.tree.code

    @staticmethod
    def from_cst(module: cst.Module, name: str, drop_comments: bool) -> "PythonModule":
        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
        src_map = dict(wrapper.resolve(PositionProvider))
        module = wrapper.module
        removed_comments = []
        if drop_comments:
            remover = CommentRemover(src_map=src_map)
            module = module.visit(remover)
            src_map = remover.src_map
            removed_comments = remover.removed_lines
            removed_comments.sort(key=lambda c: src_map[c].start.line)
        _fix_function_location_(src_map)
        return _build_python_module(module, src_map, name, removed_comments)

    def __repr__(self):
        return f"PythonModule(n_functions={len(self.functions)}, n_classes={len(self.classes)})"

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        return (e for e in self.all_elements() if isinstance(e, PythonFunction))

    def all_vars(self) -> Generator[PythonVariable, None, None]:
        return (e for e in self.all_elements() if isinstance(e, PythonVariable))

    def _all_elements(self) -> Generator[PythonElem, None, None]:
        yield from self.global_vars
        yield from self.functions
        for c in self.classes:
            yield from c.all_elements()

    def all_elements(self) -> list[PythonElem]:
        def to_tuple(pos: CodePosition):
            return (pos.line, pos.column)

        return sorted(
            self._all_elements(),
            key=lambda e: to_tuple(self.elem2pos[e.path.path].start),
        )

    def all_classes(self) -> Generator[PythonClass, None, None]:
        def rec(c: PythonClass) -> Generator[PythonClass, None, None]:
            yield c
            for sc in c.inner_classes.values():
                yield from rec(sc)

        for c in self.classes:
            yield from rec(c)

    def mask_types(self) -> "PythonModule":
        return PythonModule.from_cst(
            mask_types(self.tree), self.name, drop_comments=False
        )


@dataclass
class PythonProject:
    root_dir: Path
    modules: dict[ModuleName, PythonModule]
    symlinks: dict[ModuleName, ModuleName]
    module2src_file: dict[ModuleName, Path]

    @property
    def name(self):
        return self.root_dir.name

    def get_elem_location(self, path: ProjectPath) -> tuple[Path, CodeRange]:
        """Note that the line nubmers may differ from the orginal source file
        if there are code transformations when creating the Project. You can
        reconstruct the transformed code by calling `module.code` on the CSTModule."""

        file = self.module2src_file[path.module]
        span = self.modules[path.module].elem2pos[path.path]
        return file, span

    def verify_paths_unique(self):
        path_set = set[ProjectPath]()
        for m in self.modules.values():
            for e in m.all_elements():
                if e.path in path_set:
                    raise ValueError(f"Multiple elements with the path: {e.path}")
                path_set.add(e.path)

    @staticmethod
    def from_modules(
        root_dir: Path,
        modules: Iterable[PythonModule],
        src_map: dict[ModuleName, Path] = dict(),
    ) -> "PythonProject":
        p = PythonProject(root_dir, {m.name: m for m in modules}, dict(), src_map)
        p.verify_paths_unique()
        return p

    DefaultIgnoreDirs = {".venv", ".mypy_cache", ".git", "venv", "build"}

    @staticmethod
    def from_root(
        root: Path,
        discard_bad_files: bool = False,
        file_filter: Callable[[Path], bool] = lambda p: True,
        ignore_dirs: set[str] = DefaultIgnoreDirs,
        drop_comments: bool = True,
        text_transform: Callable[[str], str | None] = lambda s: s,
    ) -> "PythonProject":
        """
        - `root` is typically the `src/` directory or just the root of the project.
        - `src2module` is used to parse a file into a CST Module, otpionally performing
        any preprocessing transformations. The src will be discarded if this function returns None.
        """
        modules = dict()
        src_map = dict[ModuleName, Path]()
        symlinks = dict()

        if not root.exists():
            raise FileNotFoundError(f"Project root not exist: {root}")

        all_srcs = [
            f
            for f in rec_iter_files(
                root, dir_filter=lambda d: d.name not in ignore_dirs
            )
            if f.suffix == ".py" and file_filter(f)
        ]

        for src in all_srcs:
            if src.is_symlink():
                continue
            with src.open() as f:
                src_text = f.read()
            src_text = text_transform(src_text)
            if src_text is None:
                continue
            try:
                mod = cst.parse_module(src_text)
            except cst.ParserSyntaxError as e:
                if discard_bad_files:
                    continue
                raise

            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            modules[mod_name] = PythonModule.from_cst(mod, mod_name, drop_comments)
            src_map[mod_name] = src.relative_to(root)

        for src in all_srcs:
            if not src.is_symlink():
                continue
            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            origin_name = PythonProject.rel_path_to_module_name(
                src.resolve().relative_to(root)
            )
            symlinks[mod_name] = origin_name

        proj = PythonProject(root.resolve(), modules, symlinks, src_map)
        proj.verify_paths_unique()
        return proj

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        for module in self.modules.values():
            yield from module.all_funcs()

    def all_vars(self) -> Generator[PythonVariable, None, None]:
        for module in self.modules.values():
            yield from module.all_vars()

    def all_elems(self) -> Generator[PythonElem, None, None]:
        for module in self.modules.values():
            yield from module.all_elements()

    def get_sigmap(self) -> SignatureMap:
        return {e.path: e.get_signature() for e in self.all_elems()}

    def mask_types(self):
        """Replace all type annotations with `SpecialNames.TYPE_MASK`."""
        newp = copy.copy(self)
        newp.modules = {n: m.mask_types() for n, m in self.modules.items()}
        return newp

    @staticmethod
    def rel_path_to_module_name(rel_path: Path) -> ModuleName:
        parts = rel_path.parts
        assert parts[-1].endswith(".py"), f"Not a python file: {rel_path}"
        if parts[0] == "src":
            parts = parts[1:]
        if parts[-1] == "__init__.py":
            return ".".join(parts[:-1])
        else:
            # also remove the .py extension
            return ".".join([*parts[:-1], parts[-1][:-3]])


def to_abs_import_path(
    current_mod: ModuleName,
    path: str,
    allow_implicit: bool = True,
) -> Generator[ModuleName, None, None]:
    """Given the current module and an import path, return the list of modules
    (in absolute path) that import could potentially be referring to."""
    dots = 0
    while dots < len(path) and path[dots] == ".":
        dots += 1
    if dots == 0:
        yield path
        if allow_implicit:
            yield path_join(path_up(current_mod), path)
        return
    mod_segs = split_import_path(current_mod)
    assert len(mod_segs) >= dots, "Cannot go up more levels."
    result_segs = mod_segs[:-dots]
    rest = path[dots:]
    if rest:
        result_segs.append(rest)
    yield ".".join(result_segs)


@cache
def split_import_path(path: str):
    return path.split(".")


@cache
def path_up(path: str) -> str:
    segs = split_import_path(path)
    return ".".join(segs[:-1])


@cache
def path_join(path: str, subpath: str) -> str:
    return f"{path}.{subpath}" if path else subpath


@dataclass
class ProjectUsage:
    user: ProjectPath
    used: ProjectPath
    is_certain: bool  # some usage might not be certain, e.g. if it's a method call on an expression
    callsite: cst.Call | None

    def __str__(self):
        return (
            f"{self.user} {'' if self.is_certain else 'potentially '}uses {self.used}"
        )


class ModuleHierarchy:
    def __init__(self):
        self.children = dict[str, "ModuleHierarchy"]()
        # maps from implcit relative imports to the modules that they actually refer to
        self._implicit_imports: dict[tuple[ModuleName, ModuleName], ModuleName] = dict()

    def __repr__(self):
        return f"ModuleNamespace({self.children})"

    def add_module(self, segs: list[str]) -> None:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                namespace.children[s] = ModuleHierarchy()
                namespace = namespace.children[s]

    def has_module(self, segs: list[str]) -> bool:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                return False
        return True

    def resolve_path(self, segs: list[str]) -> ProjectPath | None:
        if len(segs) < 2:
            return None
        namespace = self
        matched = 0
        for s in segs[:-1]:
            if s in namespace.children:
                namespace = namespace.children[s]
                matched += 1
            else:
                break
        if matched == 0:
            return None
        return ProjectPath(".".join(segs[:matched]), ".".join(segs[matched:]))

    @staticmethod
    def from_modules(modules: Iterable[str]) -> "ModuleHierarchy":
        root = ModuleHierarchy()
        for m in modules:
            root.add_module(split_import_path(m))
        return root


def sort_modules_by_imports(
    imports: Mapping[ModuleName, set[ModuleName]]
) -> list[ModuleName]:
    "Sort modules topologically according to imports"
    sorted_modules = list[str]()
    visited = set[str]()

    def visit(m: str) -> None:
        if m in visited or m not in imports:
            return
        visited.add(m)
        if m in imports:
            for m2 in imports[m]:
                visit(m2)
        sorted_modules.append(m)

    for m in imports:
        visit(m)
    return sorted_modules


def build_project_namespaces_(
    project: PythonProject, ns_heir: ModuleHierarchy
) -> "dict[ModuleName, ProjNamespace]":
    class NsBuilder(cst.CSTVisitor):
        def __init__(
            self, module_path: str, module2ns: Mapping[ModuleName, ProjNamespace]
        ):
            self.module_path = module_path
            self.module2ns: Mapping[ModuleName, ProjNamespace] = module2ns
            self.symbol_map = dict[str, ProjectPath]()
            # self.star_imports = set[ModuleName]()
            # self.single_imports = dict[str, ProjectPath]()

        # todo: handle imported modules and renamed modules
        def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
            src_mod: ModuleName = ".".join(
                parse_module_path(
                    node.module, self.module_path, dots := len(node.relative)
                )
            )
            if src_mod not in self.module2ns:
                if dots == 0:
                    # try parse it as a relative import
                    src_mod = ".".join(
                        parse_module_path(node.module, self.module_path, 1)
                    )
                    if src_mod not in self.module2ns:
                        return
                else:
                    return

            src_ns = self.module2ns[src_mod]
            if isinstance(node.names, cst.ImportStar):
                self.symbol_map.update(src_ns)
            else:
                for name in node.names:
                    match name.name:
                        case cst.Name(value=n1) if n1 in src_ns:
                            n2 = (
                                name.asname.value
                                if isinstance(name.asname, cst.Name)
                                else n1
                            )
                            self.symbol_map[n2] = src_ns[n1]

        def visit_FunctionDef(self, node) -> Optional[bool]:
            return False

        def visit_ClassDef(self, node) -> Optional[bool]:
            return False

        def visit_Assign(self, node: "cst.Assign") -> Optional[bool]:
            return False

        def visit_AnnAssign(self, node: "cst.AnnAssign") -> Optional[bool]:
            return False

        def build(self) -> ProjNamespace:
            return persist_map(self.symbol_map)

    """Return the visible project-defined symbols in each module."""
    timed = UsageAnalysis.TLogger.timed
    with timed("sort modules"):
        # first, update the imported modules of each module by resolving
        # any implicit relative imports
        imported_modules = dict[ModuleName, set[ModuleName]]()
        for mname, mod in project.modules.items():
            base_segs = mname.split(".")
            base_segs.pop()
            additional_imports = set[ModuleName]()
            for im in imported_modules.get(mname, mod.imported_modules):
                if not ns_heir.has_module(
                    i_segs := im.split(".")
                ) and ns_heir.has_module(i_segs := base_segs + i_segs):
                    resolved = ".".join(i_segs)
                    additional_imports.add(resolved)
                    ns_heir._implicit_imports[(mname, im)] = resolved
            imported_modules[mname] = mod.imported_modules | additional_imports
        sorted_modules = sort_modules_by_imports(imported_modules)

    result = dict[ModuleName, ProjNamespace]()
    for mod in sorted_modules:
        mv = NsBuilder(mod, result)
        with timed("NSbuilder.visit"):
            project.modules[mod].tree.visit(mv)
        with timed("NSbuilder.build"):
            new_ns = mv.build()
        new_ns = new_ns.update(project.modules[mod].defined_symbols)
        result[mod] = new_ns
    return result


class UsageAnalysis:
    all_usages: list[ProjectUsage]
    path2elem: dict[ProjectPath, PythonElem]
    user2used: dict[ProjectPath, list[ProjectUsage]]
    used2user: dict[ProjectPath, list[ProjectUsage]]

    TLogger = TimeLogger()

    def get_var(self, path: ProjectPath) -> PythonVariable:
        v = self.path2elem[path]
        assert isinstance(v, PythonVariable)
        return v

    def get_func(self, path: ProjectPath) -> PythonFunction:
        v = self.path2elem[path]
        assert isinstance(v, PythonFunction)
        return v

    def __init__(
        self,
        project: PythonProject,
        mod2analysis: Mapping[ModuleName, "ModuleAnalysis"] | None = None,
        add_override_usages: bool = False,
        add_implicit_rel_imports: bool = False,
        record_type_usages: bool = False,
    ):
        timed = self.TLogger.timed

        self.project = project
        self.add_override_usages = add_override_usages
        self.add_implicit_rel_imports = add_implicit_rel_imports
        self.record_type_usages = record_type_usages

        self.ns_hier = ModuleHierarchy.from_modules(project.modules.keys())
        with timed("build_project_namespaces"):
            module2ns = build_project_namespaces_(project, self.ns_hier)
        self.sorted_modules = list(module2ns.keys())

        self.path2elem = {v.path: v for v in project.all_elems()}
        self.path2class = {
            cls.path: cls
            for mod in project.modules.values()
            for cls in mod.all_classes()
        }

        with timed("add mapping for star imports"):
            for mname, ns in module2ns.items():
                for s, p in ns.items():
                    if p in self.path2class:
                        cls = self.path2class[p]
                        self.path2class.setdefault(ProjectPath(mname, s), cls)
                    elif p in self.path2elem:
                        self.path2elem.setdefault(
                            ProjectPath(mname, s), self.path2elem[p]
                        )

        if mod2analysis is None:
            self.mod2analysis = {
                mname: ModuleAnalysis(
                    project.modules[mname], record_type_usages=self.record_type_usages
                )
                for mname in self.sorted_modules
            }
        else:
            self.mod2analysis = mod2analysis

        with timed("resolve subtyping relations"):
            self.superclass_map = superclass_map = dict[
                ProjectPath, Collection[QualifiedName]
            ]()
            for mname in self.sorted_modules:
                superclass_map.update(self.mod2analysis[mname].superclass_map)

        self.cls2members = cls2members = dict[ProjectPath, dict[str, PythonElem]]()
        all_usages = list[ProjectUsage]()

        def process_cls(cls: PythonClass):
            """add elements from parents to subclass."""
            if (cpath := cls.path) in cls2members:
                return  # already processed
            members = cls2members[cpath] = dict[str, PythonElem]()
            bases = superclass_map[cls.path]
            parents = [
                x for p in bases if (x := self.find_class(cpath.module, p)) is not None
            ]
            for parent in parents:
                process_cls(parent)
                # inherent parent attributes
                members.update(cls2members[parent.path])
            for a in cls.attributes.values():
                if self.add_override_usages and a.name in members:
                    # override usages
                    all_usages.append(
                        ProjectUsage(a.path, members[a.name].path, False, None)
                    )
                members[a.name] = a
            for m in cls.methods.values():
                if self.add_override_usages and m.name in members:
                    # override usages
                    all_usages.append(
                        ProjectUsage(m.path, members[m.name].path, False, None)
                    )
                members[m.name] = m
            for name, el in members.items():
                self.path2elem[cpath.append(name)] = el

        for mname in self.sorted_modules:
            for cls in project.modules[mname].all_classes():
                with timed("process_cls"):
                    process_cls(cls)

        all_class_members = {x.path for x in project.all_elems() if x.in_class}
        self.name2class_member = groupby(
            [self.path2elem[p] for p in all_class_members], lambda e: e.name
        )

        all_fixtures = [f for f in project.all_funcs() if f.is_fixture]
        name2fixtures = groupby(all_fixtures, lambda f: f.name)
        self.name2fixtures = {
            n: {f.path for f in fs} for n, fs in name2fixtures.items()
        }

        with timed("generate usages"):
            for mname in self.sorted_modules:
                ma = self.mod2analysis[mname]
                for caller, span, qname, call in ma.module_usages:
                    all_usages.extend(self.generate_usages(mname, caller, qname, call))

        best_usages = dict[tuple[ProjectPath, ProjectPath], ProjectUsage]()
        for u in all_usages:
            up = u.user, u.used
            if up not in best_usages or u.is_certain > best_usages[up].is_certain:
                best_usages[up] = u

        all_usages = list(best_usages.values())
        self.all_usages = all_usages
        self.user2used = groupby(all_usages, lambda u: u.user)
        self.used2user = groupby(all_usages, lambda u: u.used)

    def find_class(self, mname: ModuleName, qname: QualifiedName) -> PythonClass | None:
        cls_path = None
        match qname.source:
            case QualifiedNameSource.IMPORT:
                for abs_p in to_abs_import_path(
                    mname, qname.name, self.add_implicit_rel_imports
                ):
                    segs = split_import_path(abs_p)
                    if len(segs) >= 2:
                        cls_path = self.ns_hier.resolve_path(segs)
                        break
            case QualifiedNameSource.LOCAL:
                cls_path = ProjectPath(mname, qname.name)

        if cls_path in self.path2class:
            return self.path2class[cls_path]
        return None

    def generate_usages(
        self,
        mname: ModuleName,
        caller: ProjectPath,
        qname: QualifiedName,
        parent_call: cst.Call | None,
    ) -> Iterable[ProjectUsage]:
        def usage(used: ProjectPath, is_certain: bool):
            return ProjectUsage(caller, used, is_certain, parent_call)

        def gen_class_usages(member_name: str):
            if member_name.startswith("__") and member_name.endswith("__"):
                # skip common methods like __init__
                return
            for e in self.name2class_member.get(member_name, []):
                yield usage(e.path, is_certain=False)

        def visible_fixtures(fix_name: str) -> Generator[ProjectPath, None, None]:
            psegs = caller.path.split(".")
            # try finding in the local module
            while psegs:
                psegs.pop()
                psegs.append(fix_name)
                yield ProjectPath(caller.module, ".".join(psegs))
                psegs.pop()
            # try finding in conftest files
            msegs = mname.split(".")
            while msegs:
                msegs.pop()
                msegs.append("conftest")
                yield ProjectPath(".".join(msegs), fix_name)
                msegs.pop()

        def gen_fixture_usages(fix_name: str):
            candidates = self.name2fixtures.get(fix_name, None)
            if not candidates:
                return
            for vf in visible_fixtures(fix_name):
                if vf in candidates:
                    yield usage(vf, is_certain=True)
                    return

        def gen_constructor_usages(cls: PythonClass):
            if not parent_call:
                return
            cpath = cls.path
            used_elems = list[tuple[ProjectPath, bool]]()
            cons_path = cpath.append("__init__")
            if cons_path in self.path2elem:
                used_elems.append((cons_path, True))
            else:
                for v in cls.attributes.values():
                    used_elems.append((v.path, True))
                # maybe also used members from parent class
                for el in self.cls2members[cpath].values():
                    if isinstance(el, PythonVariable):
                        used_elems.append((el.path, False))
            for u, certain in used_elems:
                yield usage(self.path2elem[u].path, is_certain=certain)

        def resolve_local_usages(name: str):
            segs = name.split(".")
            # try resolve all usages in the access chain until a certain usage is found
            # For example, if we have `a.b.c.d` and we can resolve `a.b` to a certain element,
            # but not `a.b.c`, we will then also generate class usages for `.c` and `.d`.
            while segs:
                match segs:
                    case ["<attr>", m]:
                        # method fuzzy match case 1
                        yield from gen_class_usages(m)
                        break
                    case ["<fixture>", m]:
                        yield from gen_fixture_usages(m)
                        break
                    case [*prefix, cls, _, "<locals>", "self", m]:
                        segs = [*prefix, cls, m]

                callee = ProjectPath(mname, ".".join(segs))
                if callee in self.path2class:
                    if self.record_type_usages:
                        yield usage(callee, is_certain=True)
                    yield from gen_constructor_usages(self.path2class[callee])
                elif callee in self.path2elem:
                    yield usage(self.path2elem[callee].path, is_certain=True)
                elif len(segs) >= 2 and segs[-2] != "<locals>":
                    # method fuzzy match case 3
                    yield from gen_class_usages(segs[-1])
                    segs.pop()
                    continue
                break

        match qname.source:
            case QualifiedNameSource.IMPORT:
                for abs_p in to_abs_import_path(
                    mname, qname.name, self.add_implicit_rel_imports
                ):
                    segs = split_import_path(abs_p)
                    callee = self.ns_hier.resolve_path(segs)
                    if callee is None:
                        continue
                    if callee in self.path2class:
                        if self.record_type_usages:
                            yield usage(callee, is_certain=True)
                        yield from gen_constructor_usages(self.path2class[callee])
                        break
                    elif callee in self.path2elem:
                        yield usage(self.path2elem[callee].path, is_certain=True)
                        break
            case QualifiedNameSource.LOCAL:
                yield from resolve_local_usages(qname.name)

    def assert_usages(self, caller: str, *callees: tuple[str, bool]) -> None:
        caller_p = ProjectPath.from_str(caller)
        expect = set()
        for callee, certain in callees:
            callee_p = ProjectPath.from_str(callee)
            expect.add((callee_p, certain))

        actual = {(u.used, u.is_certain) for u in self.user2used.get(caller_p, list())}

        try:
            assert_eq(actual, expect)
        except:
            usages = self.mod2analysis[caller_p.module].module_usages
            usages = groupby(usages, lambda x: x[0]).get(caller_p, [])
            print(f"Raw callees:")
            for u in usages:
                print("\t", u[2])
            raise

    def get_stats(self) -> dict:
        usages = self.all_usages
        n_certain = sum(1 for u in usages if u.is_certain)
        n_potential = len(usages) - n_certain
        return {
            "n_usages": len(usages),
            "n_certain": n_certain,
            "n_potential": n_potential,
        }


class ModuleAnalysis:
    module: PythonModule
    node2qnames: dict[cst.CSTNode, Collection[QualifiedName]]
    node2pos: dict[cst.CSTNode, CodeRange]
    record_type_usages: bool

    def __init__(self, mod: PythonModule, record_type_usages: bool = False):
        def get_value(v):
            if isinstance(v, LazyValue):
                return v()
            return v

        self.module = mod
        wrapper = cst.MetadataWrapper(mod.tree, unsafe_skip_copy=True)
        # below need to be dict to be pickleable
        self.node2qnames = {
            k: get_value(v) for k, v in wrapper.resolve(QualifiedNameProvider).items()
        }
        self.node2pos = dict(wrapper.resolve(PositionProvider))
        self.record_type_usages = record_type_usages

    @cached_property
    def module_usages(
        self,
    ) -> Sequence[tuple[ProjectPath, CodeRange, QualifiedName, Optional[cst.Call]]]:
        """
        Compute a mapping from each method/function to the methods and functions they use.
        """

        recorder = UsageRecorder(
            self.node2qnames, self.node2pos, self.record_type_usages
        )
        result = list[tuple[ProjectPath, CodeRange, QualifiedName, cst.Call | None]]()

        for e in self.module.all_elements():
            match e:
                case PythonFunction():
                    e.tree.visit(recorder)
                    self_names = self.node2qnames[e.tree.name]

                    # generate fixture usages
                    if e.is_fixture_user:
                        for arg in e.tree.params.params:
                            fix_name = QualifiedName(
                                f"<fixture>.{arg.name.value}", QualifiedNameSource.LOCAL
                            )
                            fix_usage = (self.node2pos[arg.name], fix_name, None)
                            recorder.usages.append(fix_usage)
                case PythonVariable():
                    self_names = []
                    for a in e.assignments:
                        if a.value:
                            a.value.visit(recorder)
            # we only keep the first occurance of each qualified name to save space
            best_callee = dict[QualifiedName, tuple[cst.Call | None, CodeRange]]()
            for span, qn, parent_call in recorder.usages:
                if qn in self_names:
                    continue  # don't record self references
                if qn not in best_callee or (
                    best_callee[qn][0] is None and parent_call is not None
                ):
                    best_callee[qn] = (parent_call, span)

            for qn, (parent_call, span) in best_callee.items():
                result.append((e.path, span, qn, parent_call))
            recorder.usages.clear()

        return result

    @cached_property
    def superclass_map(self) -> Mapping[ProjectPath, Collection[QualifiedName]]:
        """Map the path of each class to its superclasses."""
        result = dict[ProjectPath, list[QualifiedName]]()
        for cls in self.module.all_classes():
            superclasses = list()
            for b in cls.tree.bases:
                if b.value in self.node2qnames and self.node2qnames[b.value]:
                    superclasses.extend(self.node2qnames[b.value])
                elif isinstance(b.value, cst.Name):
                    # unresovled parent class is treated as local for later processing
                    superclasses.append(
                        QualifiedName(b.value.value, QualifiedNameSource.LOCAL)
                    )
            result[cls.path] = superclasses
        return result


@dataclass
class SingatureTypePrediction:
    predicted: PythonType
    expected: PythonType
    path: ProjectPath
    index: int
    cat: AnnotCat


class SignatureErrorAnalysis:
    accuracies: dict[str, Any]
    errors: dict[str, list[SingatureTypePrediction]]

    def __init__(
        self,
        predictions: dict[str, SignatureMap],
        labels: dict[str, SignatureMap],
        metric: AccuracyMetric,
        error_on_mismatched_signature: bool = False,
    ):
        sig_preds = list[SingatureTypePrediction]()
        pred_project = list[str]()
        n_labels = 0
        n_skipped = 0
        n_missing = [0]

        def match_signatures(
            project: str, path: ProjectPath, p_sig: ElemSignature, l_sig: ElemSignature
        ):
            def record_pair(
                pred: cst.Annotation | None,
                label: cst.Annotation | None,
                cat: AnnotCat,
                pos: int,
            ):
                if label is None or (lt := parse_type_expr(label.annotation)) is None:
                    # no label available
                    return
                if pred is None:
                    if error_on_mismatched_signature:
                        raise RuntimeError(
                            f"Prediction missing at position {pos}. label: {l_sig}, pred: {p_sig}"
                        )
                    else:
                        n_missing[0] += 1
                        return
                assert pred is not None
                pt = parse_type_expr(pred.annotation, silent=True)
                if pt is None:
                    pt = PythonType.from_name("Any")
                sig_pred = SingatureTypePrediction(
                    predicted=pt,
                    expected=lt,
                    path=path,
                    index=pos,
                    cat=cat,
                )
                pred_project.append(project)
                sig_preds.append(sig_pred)

            match p_sig, l_sig:
                case (VariableSignature(pa), VariableSignature(la, in_class=in_class)):
                    cat = AnnotCat.ClassAtribute if in_class else AnnotCat.GlobalVar
                    record_pair(pa, la, cat, 0)
                case (
                    FunctionSignature(p_params, p_return),
                    FunctionSignature(l_params, l_return, in_class=in_class),
                ):
                    for i, n in enumerate(l_params.keys()):
                        record_pair(
                            p_params.get(n, None), l_params[n], AnnotCat.FuncArg, i
                        )
                    record_pair(p_return, l_return, AnnotCat.FuncReturn, len(l_params))
                case _:
                    if error_on_mismatched_signature:
                        raise RuntimeError(
                            f"Mismatched signatures: label={l_sig}, pred={p_sig}"
                        )
                    for i, (cat, a) in enumerate(l_sig.get_annotated()):
                        record_pair(None, a, cat, i)

        for project in labels:
            l_map = labels[project]
            p_map = predictions[project]
            for path in l_map:
                l_sig = l_map[path]
                n_labels += l_sig.n_annotated()
                if (p_sig := p_map.get(path)) is not None:
                    try:
                        match_signatures(project, path, p_sig, l_sig)
                    except:
                        print(f"In project: {project}, element: {path}")
                        raise
                else:
                    n_skipped += l_sig.n_annotated()

        incorrect_set = list[int]()
        accs = type_accuracies(
            [p.predicted for p in sig_preds],
            [p.expected for p in sig_preds],
            [p.cat for p in sig_preds],
            metric,
            output_incorrect_set=incorrect_set,
        )
        if n_skipped > 0:
            accs["n_skipped_types"] = n_skipped
        if n_missing[0] > 0:
            accs["n_missing_types"] = n_missing[0]

        self.accuracies = accs

        all_errors: dict[str, list[SingatureTypePrediction]] = {
            p: [] for p in predictions
        }
        for i in incorrect_set:
            sig_pred = sig_preds[i]
            all_errors[pred_project[i]].append(sig_pred)

        self.errors = all_errors


# -----------------------------------------------------------
# utilities for static analysis


class _VisitKind(enum.Enum):
    Root = enum.auto()
    Class = enum.auto()
    Function = enum.auto()


def _fix_function_location_(node2range: dict[cst.CSTNode, CodeRange]):
    """
    Change the start code range of a function to point to its first decorator,
    as opposed to the `def` statement.
    """
    for node in node2range:
        if isinstance(node, cst.FunctionDef) and node.decorators:
            dec_range = node2range[node.decorators[0]]
            node2range[node] = CodeRange(
                start=dec_range.start,
                end=node2range[node].end,
            )


def _build_python_module(
    module: cst.Module,
    node2location: Mapping[cst.CSTNode, CodeRange],
    module_name: ModuleName,
    removed_comments: list[cst.CSTNode],
):
    """Construct a `PythonModule` from a `cst.Module`.
    If multiple definitions of the same name are found, only the last one is kept."""
    imported_modules = set[str]()
    defined_symbols = dict[str, ProjectPath]()
    elem2pos = dict[ElemPath, CodeRange]()

    class ModuleBuilder(cst.CSTVisitor):
        def __init__(self):
            super().__init__()
            self.functions = dict[str, PythonFunction]()
            self.global_vars = dict[str, PythonVariable]()
            self.classes = dict[str, PythonClass]()
            self.module_base = ProjectPath(module_name, "")
            self.class_stack = list[PythonClass]()
            self.visit_stack = [_VisitKind.Root]

        @property
        def current_class(self) -> PythonClass | None:
            return self.class_stack[-1] if self.class_stack else None

        @property
        def current_path(self):
            if self.current_class is not None:
                return self.current_class.path
            else:
                return self.module_base

        def _record_elem(self, e: PythonElem, location_node: cst.CSTNode):
            cls = self.current_class
            vars = cls.attributes if cls else self.global_vars
            funcs = cls.methods if cls else self.functions
            classes = cls.inner_classes if cls else self.classes
            is_new_def = False

            classes.pop(e.name, None)
            if cls is None:
                defined_symbols[e.name] = e.path
            match e:
                case PythonFunction(n):
                    vars.pop(n, None)
                    funcs[n] = e
                    is_new_def = True
                case PythonVariable(
                    n, assignments=assignments, wrapped_assignments=wrapped_assignments
                ):
                    funcs.pop(n, None)
                    if n in vars:
                        assert_eq(vars[n].path, e.path)
                        vars[n].assignments.extend(assignments)
                        vars[n].wrapped_assignments.extend(wrapped_assignments)
                    else:
                        vars[n] = e
                        is_new_def = True
                case _:
                    raise NotImplementedError()
            if is_new_def:
                crange = node2location[location_node]
                elem2pos[e.path.path] = crange

        def visit_FunctionDef(self, node: cst.FunctionDef):
            parent_type = self.visit_stack[-1]
            self.visit_stack.append(_VisitKind.Function)
            if parent_type == _VisitKind.Function:
                # skip inner functions
                return False
            for dec in node.decorators:
                if get_decorator_name(dec) == "overload":
                    return False

            name = node.name.value
            func = PythonFunction(
                name=node.name.value,
                path=self.current_path.append(name),
                tree=node,
                parent_class=self.current_class.path if self.current_class else None,
            )
            self._record_elem(func, node)

        def leave_FunctionDef(self, node) -> None:
            assert self.visit_stack[-1] == _VisitKind.Function
            self.visit_stack.pop()

        def visit_ClassDef(self, node: cst.ClassDef):
            parent_type = self.visit_stack[-1]
            parent_cls = self.current_class
            cls = PythonClass(
                name=node.name.value,
                path=self.current_path.append(node.name.value),
                attributes=dict(),
                methods=dict(),
                inner_classes=dict(),
                tree=node,
                parent_class=parent_cls.path if parent_cls else None,
            )
            if parent_cls:
                parent_cls.inner_classes[cls.name] = cls
            self.visit_stack.append(_VisitKind.Class)
            self.class_stack.append(cls)
            if parent_type == _VisitKind.Root:
                self.global_vars.pop(cls.name, None)
                self.functions.pop(cls.name, None)
                self.classes[cls.name] = cls
                defined_symbols[cls.name] = cls.path

        def leave_ClassDef(self, node: cst.ClassDef):
            assert self.visit_stack[-1] == _VisitKind.Class
            self.class_stack.pop()
            self.visit_stack.pop()

        # record global_vars and class attributes
        def visit_AnnAssign(self, node: cst.AnnAssign):
            cls = self.current_class
            cls_path = cls.path if cls else None
            var = None
            match self.visit_stack[-1], node.target:
                case (_VisitKind.Root, cst.Name(value=n) as tree):
                    # global var assignment
                    var = PythonVariable(
                        n, ProjectPath(module_name, n), None, tree, [node], []
                    )
                case (_VisitKind.Class, cst.Name(value=n) as tree) if cls:
                    # initialized outside of methods
                    var = PythonVariable(
                        n, cls.path.append(n), cls_path, tree, [node], []
                    )
                case (
                    _VisitKind.Function,
                    cst.Attribute(
                        value=cst.Name(value="self"), attr=cst.Name(value=n)
                    ) as tree,
                ) if cls:
                    # initialized/accessed inside methods
                    var = PythonVariable(
                        n, cls.path.append(n), cls_path, tree, [], [node]
                    )
            if var is not None:
                self._record_elem(var, node)
            return False

        # record global_vars and class attributes
        def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
            cls = self.current_class
            cls_path = cls.path if cls else None

            # class member declaration
            for target in node.targets:
                var = None
                match self.visit_stack[-1], target.target:
                    case (_VisitKind.Root, cst.Name(value=n) as tree):
                        # global var assignment
                        var = PythonVariable(
                            n, ProjectPath(module_name, n), None, tree, [node], []
                        )
                    case (_VisitKind.Class, cst.Name(value=n) as tree) if cls:
                        # initialized outside of methods
                        var = PythonVariable(
                            n, cls.path.append(n), cls_path, tree, [node], []
                        )
                    case (
                        _VisitKind.Function,
                        cst.Attribute(
                            value=cst.Name(value="self"), attr=cst.Name(value=n)
                        ) as tree,
                    ) if cls:
                        # initialized/accessed inside methods
                        var = PythonVariable(
                            n, cls.path.append(n), cls_path, tree, [], [node]
                        )
                if var is not None:
                    self._record_elem(var, node)
            return False

        def visit_Import(self, node: cst.Import):
            for alias in node.names:
                segs = list[str]()
                for seg in parse_module_path(alias.name, module_name, 0):
                    segs.append(seg)
                    imported_modules.add(".".join(segs))
            return False

        def visit_ImportFrom(self, node: cst.ImportFrom):
            segs = list[str]()
            for seg in parse_module_path(node.module, module_name, len(node.relative)):
                segs.append(seg)
                imported_modules.add(".".join(segs))
            prefix = ".".join(segs)
            if not isinstance(node.names, cst.ImportStar):
                for alias in node.names:
                    posfix = [prefix]
                    # modules could also be imported via import from statements
                    for seg in parse_module_path(alias.name, "", 0):
                        posfix.append(seg)
                    imported_modules.add(".".join(posfix))
            return False

    builder = ModuleBuilder()
    module.visit(builder)
    src_map = node2location
    if not isinstance(src_map, dict):
        src_map = dict(node2location)
    return PythonModule(
        global_vars=list(builder.global_vars.values()),
        functions=list(builder.functions.values()),
        classes=list(builder.classes.values()),
        name=module_name,
        imported_modules=imported_modules,
        defined_symbols=defined_symbols,
        tree=module,
        elem2pos=elem2pos,
        location_map=src_map,
        removed_comments=removed_comments,
    )


def parse_module_path(
    path_ex: cst.Attribute | cst.Name | None, cur_mod: str, dots: int
) -> list[str]:
    result = list[str]() if dots == 0 else cur_mod.split(".")[:-dots]

    def rec(ex):
        match ex:
            case None:
                pass
            case cst.Name(value=name):
                result.append(name)
            case cst.Attribute(value=attr, attr=cst.Name(value=name)):
                rec(attr)
                result.append(name)
            case _:
                raise ValueError(f"Cannot parse {ex} as module path")

    rec(path_ex)
    return result


class UsageRecorder(cst.CSTVisitor):
    """
    Records the (partially resoved) symbol usages.
    """

    def __init__(
        self,
        name_mapping: Mapping[cst.CSTNode, Collection[QualifiedName]],
        span_mapping,
        record_type_usages: bool,
    ):
        super().__init__()

        self.name_mapping = name_mapping
        self.span_mapping = span_mapping
        self.record_type_usages = record_type_usages
        self.parents = list[cst.CSTNode]()
        self.usages = list[tuple[CodeRange, QualifiedName, cst.Call | None]]()

    def resolve_ref(self, name: cst.CSTNode) -> list[QualifiedName]:
        "Return a list of qualified names that the given name could refer to."
        if is_access_chain(name) and name in self.name_mapping:
            srcs = self.name_mapping[name]
            if len(srcs) == 0 and isinstance(name, cst.Name):
                # unresolved symbols are put into the 'LOCAL' category for later processing
                # due to star imports.
                return [QualifiedName(name.value, QualifiedNameSource.LOCAL)]
            else:
                return [s for s in srcs if s.name != "builtins.None"]
        return []

    def parent_call(self) -> cst.Call | None:
        match self.parents:
            case [*_, cst.Call() as c]:
                return c
            case _:
                return None

    def record_name_use(self, name: cst.CSTNode):
        for src in self.resolve_ref(name):
            self.usages.append((self.span_mapping[name], src, self.parent_call()))

    def visit_Attribute(self, node: cst.Attribute):
        if not self.resolve_ref(node):
            # if the access cannot be resolved (e.g., is an expression), we record
            # the usage as potential method access.
            qname = QualifiedName(
                f"<attr>.{node.attr.value}", QualifiedNameSource.LOCAL
            )
            span = self.span_mapping[node]
            self.usages.append((span, qname, self.parent_call()))
            return True
        else:
            # if this access is resolved, do not record remaining prefixes as usages
            return False

    def on_visit(self, node: cst.CSTNode) -> Optional[bool]:
        self.record_name_use(node)
        self.parents.append(node)
        return super().on_visit(node)

    def on_leave(self, node: cst.CSTNode) -> Optional[bool]:
        self.parents.pop()

    def visit_Annotation(self, node: cst.Annotation):
        return self.record_type_usages


@lru_cache(maxsize=128)
def is_access_chain(node: cst.CSTNode) -> bool:
    """Return whether the node is an access chain of the form `a.b.c`. A simple name
    is also considered an access chain."""
    match node:
        case cst.Attribute(lhs, cst.Name()):
            return is_access_chain(lhs)
        case cst.Name():
            return True
        case _:
            return False


def as_access_chain(node: cst.CSTNode) -> list[str] | None:
    result = list[str]()
    while True:
        match node:
            case cst.Attribute(lhs, cst.Name(value=n)):
                result.append(n)
                node = lhs
            case cst.Name(value=n):
                result.append(n)
                break
            case _:
                return None
    result.reverse()
    return result


def stub_from_module(
    m: cst.Module,
    lightweight=True,
    rm_comments=True,
    rm_imports=True,
    keep_types=False,
) -> cst.Module:
    """Generate a stub module from normal python code."""
    if rm_comments:
        m = remove_comments(m)
    if rm_imports:
        m, _ = remove_imports(m)
    if lightweight:
        m = m.visit(LightStubGenerator(keep_types=keep_types))
    else:
        m = m.visit(StubGenerator(keep_types=keep_types))
    m = remove_empty_lines(m)
    return m


@dataclass
class ClassNamespace:
    all_elems: set[str] = field(default_factory=set)
    declared_elems: set[str] = field(default_factory=set)


class LightStubGenerator(cst.CSTTransformer):
    """Generate a light-weight stub module from a Python module.
    Only class headers and global assignements involving potential type aliases are kept.
    """

    OMIT = cst.SimpleStatementLine([cst.Expr(cst.Ellipsis())])

    def __init__(self, keep_types: bool):
        self.nest_level = 0
        self.keep_types = keep_types

    def visit_ClassDef(self, node: cst.ClassDef):
        self.nest_level += 1

    def leave_ClassDef(self, node, updated: cst.ClassDef):
        self.nest_level -= 1
        return updated.with_changes(body=cst.IndentedBlock([self.OMIT]))

    def visit_FunctionDef(self, node):
        self.nest_level += 1

    def leave_FunctionDef(self, node, updated: cst.FunctionDef):
        self.nest_level -= 1
        return cst.RemoveFromParent()

    def leave_Annotation(self, node, updated: cst.Annotation):
        if self.keep_types:
            return updated
        return updated.with_changes(annotation=cst.Ellipsis())

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        return cst.RemoveFromParent()

    def leave_Assign(self, node, updated: cst.Assign):
        match updated:
            case cst.Assign(targets=targets, value=rhs) if self.nest_level == 0:
                if all(map(is_type_lhs, targets)) and is_type_rhs(rhs):
                    return updated
        return cst.RemoveFromParent()

    def leave_Decorator(self, node, updated: cst.Decorator):
        # omit decorator call arguments
        match updated.decorator:
            case cst.Call(func=f):
                new_call = cst.Call(f, [cst.Arg(cst.Ellipsis())])
                updated = updated.with_changes(decorator=new_call)
        return updated


def is_type_lhs(target: cst.AssignTarget):
    return isinstance(target.target, cst.Name)


_TypeDeclareRHS = {"TypeVar", "NamedTuple", "namedtuple"}


def is_type_rhs(expr: cst.BaseExpression):
    # A type-declaring rhs can only be one of the following:
    # - a simple type, e.g., A = a.Foo
    # - a generic type, e.g., A = Foo[str, T]
    # - a type var, e.g., A = TypeVar("A")
    # - a namedtuple, e.g., Foo = namedtuple('Foo', ['x', 'y'])
    match expr:
        case _ if is_access_chain(expr):
            return True
        case cst.Subscript(value=value):
            return is_access_chain(value)
        case cst.Call(
            func=cst.Name(value=name) | cst.Attribute(attr=cst.Name(value=name))
        ) if name in _TypeDeclareRHS:
            return True
        case _:
            return False


def mask_assign_type(node: cst.AnnAssign) -> cst.Assign | cst.AnnAssign:
    if node.value is not None:
        return cst.Assign([cst.AssignTarget(node.target)], node.value)
    else:
        return node


class StubGenerator(cst.CSTTransformer):
    """Generate a stub module from a Python module."""

    OMIT = cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])

    def __init__(self, keep_types: bool):
        self.keep_types = keep_types
        self.ns_stack = list[ClassNamespace]()
        self.nest_level = 0

    def register_elem(self, name: str, declared: bool):
        if self.ns_stack:
            s = self.ns_stack[-1]
            s.all_elems.add(name)
            if declared:
                s.declared_elems.add(name)

    def visit_ClassDef(self, node: cst.ClassDef):
        self.nest_level += 1
        self.ns_stack.append(ClassNamespace())

    def leave_ClassDef(self, node, updated: cst.ClassDef):
        s = self.ns_stack.pop()
        to_declare = s.all_elems.difference(s.declared_elems)
        if to_declare:
            more_stmts = [cst.parse_statement(f"{n}: ...") for n in to_declare]
            new_stmts = list(updated.body.body) + more_stmts
            updated = updated.with_changes(
                body=updated.body.with_changes(body=new_stmts)
            )
        self.nest_level -= 1
        return updated

    def visit_FunctionDef(self, node):
        self.nest_level += 1

    def leave_FunctionDef(self, node, updated: cst.FunctionDef):
        self.register_elem(updated.name.value, True)
        self.nest_level -= 1
        return updated.with_changes(body=StubGenerator.OMIT, returns=None)

    def leave_Annotation(self, node, updated: cst.Annotation):
        if self.keep_types:
            return updated
        return updated.with_changes(annotation=cst.Ellipsis())

    def leave_Param(self, node, updated: cst.Param):
        # remove parameter type annotation and default value
        if self.keep_types:
            return updated
        if updated.default is not None:
            updated = updated.with_changes(default=cst.Ellipsis())
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        # omit rhs of annotated assignments (if any)
        if updated.value is not None:
            updated = updated.with_changes(value=cst.Ellipsis())
        if self.keep_types:
            return updated
        return mask_assign_type(updated)

    def leave_Assign(self, node, updated: cst.AnnAssign):
        return updated.with_changes(value=cst.Ellipsis())

    def leave_Attribute(self, node, updated: cst.Assign):
        # record all atribute accesses involving `self`
        match updated:
            case cst.Attribute(
                value=cst.Name(value="self"),
                attr=cst.Name(value=elem_name),
            ):
                self.register_elem(elem_name, False)
        return updated

    def leave_Decorator(self, node, updated: cst.Decorator):
        # omit decorator call arguments
        match updated.decorator:
            case cst.Call(func=f):
                new_call = cst.Call(f, [cst.Arg(cst.Ellipsis())])
                updated = updated.with_changes(decorator=new_call)

        return updated


class EmptyLineRemove(cst.CSTTransformer):
    def on_leave(self, node, updated):
        if hasattr(updated, "leading_lines") and updated.leading_lines:
            return updated.with_changes(leading_lines=[])
        return updated


class CommentRemover(cst.CSTTransformer):
    def __init__(self, src_map: Mapping[cst.CSTNode, CodeRange] | None = None):
        super().__init__()
        self.removed_lines = list[cst.CSTNode]()
        self.src_map = dict(src_map) if src_map else dict()

    def on_leave(self, original: cst.CSTNode, updated: cst.CSTNode):
        result = super().on_leave(original, updated)
        if (
            isinstance(result, cst.CSTNode)
            and (old_span := self.src_map.pop(original, None)) is not None
        ):
            self.src_map[result] = old_span
        return result

    def leave_IndentedBlock(
        self, node: cst.IndentedBlock, updated: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        to_keep = []
        for n in updated.body:
            if self.is_doc_string(n):
                self.removed_lines.append(n)
            else:
                to_keep.append(n)
        if len(to_keep) != len(updated.body):
            body = type(updated.body)(to_keep)  # type: ignore
            return updated.with_changes(body=body)
        else:
            return updated

    def leave_Module(self, node, updated):
        return self.leave_IndentedBlock(node, updated)

    def leave_EmptyLine(self, node: cst.EmptyLine, updated: cst.EmptyLine):
        if updated.comment is not None:
            self.removed_lines.append(updated.comment)
            return cst.RemoveFromParent()
        else:
            return updated

    def leave_TrailingWhitespace(self, node, updated: cst.TrailingWhitespace):
        if updated.comment is not None:
            return updated.with_changes(
                comment=None, whitespace=cst.SimpleWhitespace("")
            )
        else:
            return updated

    def line_map(self, post_node: cst.CSTNode) -> dict[int, int]:
        """Map lines in the post-removal code into pre-removal lines.
        Line numbers are relative to the start of `node`, 0-based."""

        origin_span = self.src_map[post_node]
        start_line = origin_span.start.line
        origin_lines = origin_span.end.line - start_line + 1
        is_removed = [False for _ in range(origin_lines)]
        for c in self.removed_lines:
            span = self.src_map[c]
            for post_i in range(span.start.line, span.end.line + 1):
                is_removed[post_i - start_line] = True

        line_map = dict[int, int]()
        pre_i = 0
        for post_i, r in enumerate(is_removed):
            line_map[pre_i] = post_i
            if not r:
                pre_i += 1
        return line_map

    @staticmethod
    def is_doc_string(node: cst.BaseStatement) -> bool:
        match node:
            case cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString())]):
                return True
            case _:
                return False


class ImportsRemover(cst.CSTTransformer):
    """Removes all top-level import statements and collect them into `self.import_stmts`."""

    def __init__(self):
        self.import_stmts = set[cst.Import | cst.ImportFrom]()

    def leave_Import(self, node: cst.Import, updated: cst.Import):
        self.import_stmts.add(updated)
        return cst.RemoveFromParent()

    def leave_ImportFrom(self, node: cst.ImportFrom, updated: cst.ImportFrom):
        self.import_stmts.add(updated)
        return cst.RemoveFromParent()

    def visit_FunctionDef(self, node):
        # stops traversal at inner levels.
        return False

    def visit_ClassDef(self, node) -> Optional[bool]:
        return False


class AnnotRemover(cst.CSTTransformer):
    """Removes all type annotations when possible or replace them with a special symbol."""

    def __init__(self, type_mask: str = "..."):
        super().__init__()
        self.type_mask = cst.Ellipsis() if type_mask == "..." else cst.Name(type_mask)

    def leave_FunctionDef(self, node, updated: cst.FunctionDef) -> cst.FunctionDef:
        return updated.with_changes(returns=None)

    def leave_Param(self, node, updated: cst.Param) -> cst.Param:
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        updated = updated.with_changes(annotation=cst.Annotation(self.type_mask))
        return mask_assign_type(updated)


def guess_src_root(proj_root: Path):
    if (proj_root / "src").exists():
        return proj_root / "src"
    return proj_root
