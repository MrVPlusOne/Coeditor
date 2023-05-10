import traceback
from functools import wraps

from jsonrpcserver import Error, Success, method, serve

from coeditor.common import *
from coeditor.model import RetrievalEditorModel
from coeditor.service import (
    ChangeDetector,
    DecodingArgs,
    EditPredictionService,
    ServiceResponse,
)


class LazyVal(Generic[T1]):
    def __init__(self, task: Callable[[], T1], tag: int):
        self._finished = False
        self._task = task
        self.id = tag

    def get(self) -> T1:
        if not self._finished:
            assert self._task is not None
            v = self._task()
            self._task = None
            self._finished = True
            self._result = v
        return self._result


def start_server(device, port: int, print_stats: bool = True):
    model_path = "MrVPlusOne/coeditor-perm2k-base-v1.7.3"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path}' loaded on device:", device)
    dec_args = DecodingArgs(do_sample=False, num_beams=4)

    services = dict[Path, EditPredictionService]()
    tasks = dict[Path, LazyVal[ServiceResponse]]()

    def handle_error(f, *args, **kwargs):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                traceback.print_exception(e)
                return Error(code=1, message=repr(e))

        return wrapper

    @method
    @handle_error
    def initialize(project: str):
        target_dir = Path(project).resolve()

        if target_dir not in services:
            with timed_action(f"Create service for project: {target_dir}"):
                detector = ChangeDetector(target_dir)
                services[target_dir] = EditPredictionService(
                    detector,
                    model,
                    dec_args=dec_args,
                )

        return Success("OK")

    @method
    @handle_error
    def submit_problem(
        time: int, project: str, file: str, lines: Sequence[int] | int, writeLogs: bool
    ):
        initialize(project)
        target_dir = Path(project).resolve()
        service = services[target_dir]

        print(f"Suggesting edit for lines {lines} in {file}")
        path = Path(file)
        if Path.is_absolute(path):
            path = path.relative_to(target_dir)
        path = to_rel_path(path)

        service.tlogger.clear()
        log_dir = service.project / ".coeditor_logs" if writeLogs else None
        region, f = service._suggest_edit_two_steps(path, lines, log_dir)
        if target_dir in tasks and tasks[target_dir].id > time:
            return Success("Skipped")
        tasks[target_dir] = LazyVal(f, time)
        return Success(region.target_lines)

    @method
    @handle_error
    def get_result(time: int, project: str):
        target_dir = Path(project).resolve()
        cont = tasks[target_dir]
        if cont.id > time:
            return Success("Skipped")
        response = cont.get()
        service = services[target_dir]
        if print_stats:
            print("Runtime stats:")
            display(service.tlogger.as_dataframe())

        return Success(response.to_json())

    print(f"Starting suggestion server at localhost:{port}")
    serve("localhost", port)


if __name__ == "__main__":
    start_server("cuda", port=5042)
