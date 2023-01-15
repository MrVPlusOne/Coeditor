from os import PathLike
from spot.static_analysis import ModuleName, ProjectPath, PythonProject
from .common import *
import jedi, parso
from parso.python import tree
from jedi.api import helpers, convert_names, classes
from jedi import cache

# old_clear = cache.clear_time_caches
# cache.clear_time_caches = lambda: None  # turn off time cache


@dataclass
class JediUsageAnalysis:
    project: PythonProject

    def __post_init__(self):
        self.jproj = jedi.Project(self.project.root_dir)
        self.errors = dict[str, int]()
        self.tlogger: TimeLogger = TimeLogger()

    def get_module_usages(self, module: ModuleName, follow_imports: bool = True):
        src_path = self.project.module2src_file[module]
        script = jedi.Script(path=src_path, project=self.jproj)
        jmod: tree.Module = script._module_node
        usage_map = dict[tree.Name, list]()
        all_names = [
            name for k, names in jmod.get_used_names()._dict.items() for name in names
        ]
        all_names.sort(key=lambda x: x.start_pos)
        errors = self.errors
        for name in tqdm(all_names, f"Analyzing {module}"):
            name: tree.Name
            if name.value == "self":
                continue
            try:
                defs = fast_goto(
                    script,
                    name,
                    follow_imports=follow_imports,
                    follow_builtin_imports=False,
                )
                if defs:
                    usage_map[name] = defs
            except (AttributeError, AssertionError) as e:
                text = str(e)
                errors[text] = errors.setdefault(text, 0) + 1
        return usage_map

    def get_all_usages(self, follow_imports: bool = True):
        mod2usages = dict[ModuleName, dict[tree.Name, list]]()
        for module in self.project.modules:
            with self.tlogger.timed(f"get_module_usages({module})"):
                mod2usages[module] = self.get_module_usages(
                    module, follow_imports=follow_imports
                )
        print("total usages:", sum(len(us) for us in mod2usages.values()))
        return mod2usages


def fast_goto(
    script: jedi.Script,
    tree_name: tree.Name,
    *,
    follow_imports=False,
    follow_builtin_imports=False,
    only_stubs=False,
    prefer_stubs=False,
):
    """
    Goes to the name that defined the object under the cursor. Optionally
    you can follow imports.
    Multiple objects may be returned, depending on an if you can have two
    different versions of a function.

    :param follow_imports: The method will follow imports.
    :param follow_builtin_imports: If ``follow_imports`` is True will try
        to look up names in builtins (i.e. compiled or extension modules).
    :param only_stubs: Only return stubs for this method.
    :param prefer_stubs: Prefer stubs to Python objects for this method.
    :rtype: list of :class:`.Name`
    """
    name = script._get_module_context().create_name(tree_name)

    # Make it possible to goto the super class function/attribute
    # definitions, when they are overwritten.
    names = []
    if name.tree_name.is_definition() and name.parent_context.is_class():
        class_node = name.parent_context.tree_node
        class_value = script._get_module_context().create_value(class_node)
        mro = class_value.py__mro__()
        next(mro)  # Ignore the first entry, because it's the class itself.
        for cls in mro:
            names = cls.goto(tree_name.value)
            if names:
                break

    if not names:
        names = list(name.goto())

    if follow_imports:
        names = helpers.filter_follow_imports(names, follow_builtin_imports)
    names = convert_names(
        names,
        only_stubs=only_stubs,
        prefer_stubs=prefer_stubs,
    )

    defs = [classes.Name(script._inference_state, d) for d in set(names)]
    # Avoid duplicates
    return list(set(helpers.sorted_definitions(defs)))
