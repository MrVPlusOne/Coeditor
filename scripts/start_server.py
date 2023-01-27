import traceback

from jsonrpcserver import Error, InvalidParams, Result, Success, method, serve

from coeditor.api import (
    BatchArgs,
    ChangeDetectionConfig,
    DecodingArgs,
    EditPredictionService,
    QueryRefEditEncoder,
)
from coeditor.common import *
from coeditor.model import AttentionMode, RetrievalEditorModel


def start_server(
    device, port: int, drop_comments: bool = False, print_stats: bool = True
):
    # this newer model is trained with comments
    model_path = "MrVPlusOne/coeditor-xl-bi-request-stub-comments-v4"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path}' loaded on device:", device)
    batch_args = BatchArgs.service_default()
    services = dict[Path, EditPredictionService]()

    @method
    def suggestEdits(project: str, file: str, line: int, writeLogs: bool):
        target_dir = Path(project).resolve()
        if (service := services.get(target_dir)) is None:
            service = EditPredictionService(
                target_dir,
                model,
                batch_args=batch_args,
                encoder=QueryRefEditEncoder(
                    max_ref_tks=batch_args.max_ref_tks,
                    max_query_tks=batch_args.max_query_tks,
                    max_output_tks=batch_args.max_output_tks,
                ),
                dec_args=DecodingArgs(do_sample=False, num_beams=8),
                # dec_args=DecodingArgs(
                #     do_sample=True, top_p=0.95, marginalize_samples=20
                # ),
                config=ChangeDetectionConfig(drop_comments=drop_comments),
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
