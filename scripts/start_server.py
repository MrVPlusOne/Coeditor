import traceback

from jsonrpcserver import Error, InvalidParams, Result, Success, method, serve

from coeditor.common import *
from coeditor.model import AttentionMode, RetrievalEditorModel
from coeditor.service import (
    BatchArgs,
    ChangeDetector,
    DecodingArgs,
    EditPredictionService,
)


def start_server(device, port: int, print_stats: bool = True):
    # this newer model is trained with comments
    model_path = "MrVPlusOne/coeditor-xl-c3-dropout-v1.4"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path}' loaded on device:", device)
    dec_args = DecodingArgs(do_sample=False, num_beams=4)

    services = dict[Path, EditPredictionService]()

    @method
    def suggestEdits(
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
        if not Path.is_absolute(path):
            path = target_dir / path
        try:
            service.tlogger.clear()
            log_dir = service.project / ".coeditor_logs" if writeLogs else None
            response = service.suggest_edit(path, lines, log_dir)
            if print_stats:
                print("Runtime stats:")
                display(service.tlogger.as_dataframe())
            return Success(response.to_json())
        except Exception as e:
            print("Failed with exception:")
            traceback.print_exception(e)
            return Error(code=1, message=repr(e))

    print(f"Starting suggestion server at localhost:{port}")
    serve("localhost", port)


if __name__ == "__main__":
    start_server("cuda", port=5042)
