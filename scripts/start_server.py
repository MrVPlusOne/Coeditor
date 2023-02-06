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


def start_server(
    device, port: int, drop_comments: bool = False, print_stats: bool = True
):
    # this newer model is trained with comments
    model_path = "MrVPlusOne/coeditor-xl-c3-dropout-v1.4"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path}' loaded on device:", device)
    batch_args = BatchArgs.service_default()
    dec_args = DecodingArgs(do_sample=False, num_beams=4, length_penalty=0.0)

    services = dict[Path, EditPredictionService]()

    @method
    def suggestEdits(project: str, file: str, line: int, writeLogs: bool):
        target_dir = Path(project).resolve()
        if (service := services.get(target_dir)) is None:
            detector = ChangeDetector(target_dir)
            service = EditPredictionService(
                detector,
                model,
                batch_args=batch_args,
                dec_args=dec_args,
            )
            print(f"Service created for project: {target_dir}")
            services[target_dir] = service

        print(f"Suggesting edit for line {line} in {file}")
        path = Path(file)
        if not Path.is_absolute(path):
            path = target_dir / path
        try:
            service.tlogger.clear()
            model.tlogger = service.tlogger
            log_dir = service.project / ".coeditor_logs" if writeLogs else None
            response = service.suggest_edit(path, line, log_dir)
            if print_stats:
                print("Runtime stats:")
                display(service.tlogger.as_dataframe())
            return Success(response.to_json())
        except Exception as e:
            print("Failed with exception:")
            traceback.print_exception(e)
            return Error(code=1, message=repr(e))

    print(f"Starting suggestion server ({drop_comments=}) at localhost:{port}")
    serve("localhost", port)


if __name__ == "__main__":
    start_server("cuda", port=5042)
