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


def start_server(device, port: int, print_stats: bool = True):
    # this newer model is trained with comments
    model_path = "MrVPlusOne/coeditor-xl-c3-dropout-v1.4"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path}' loaded on device:", device)
    dec_args = DecodingArgs(do_sample=False, num_beams=4)

    services = dict[Path, EditPredictionService]()
    continuations = dict[Path, Callable[[], ServiceResponse]]()

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
    def submit_problem(
        project: str, file: str, lines: Sequence[int] | int, writeLogs: bool
    ):
        target_dir = Path(project).resolve()
        if (service := services.get(target_dir)) is None:
            with timed_action(f"Create service for project: {target_dir}"):
                detector = ChangeDetector(target_dir)
                service = EditPredictionService(
                    detector,
                    model,
                    dec_args=dec_args,
                )
                services[target_dir] = service

        print(f"Suggesting edit for lines {lines} in {file}")
        path = Path(file)
        if Path.is_absolute(path):
            path = path.relative_to(target_dir)
        path = to_rel_path(path)

        service.tlogger.clear()
        log_dir = service.project / ".coeditor_logs" if writeLogs else None
        region, cont = service._suggest_edit_two_steps(path, lines, log_dir)
        continuations[target_dir] = cont
        return Success(region.target_lines)

    @method
    @handle_error
    def get_result(project: str):
        target_dir = Path(project).resolve()
        f = continuations.pop(target_dir)
        response = f()
        service = services[target_dir]
        if print_stats:
            print("Runtime stats:")
            display(service.tlogger.as_dataframe())

        return Success(response.to_json())

    print(f"Starting suggestion server at localhost:{port}")
    serve("localhost", port)


if __name__ == "__main__":
    start_server("cuda", port=5042)
