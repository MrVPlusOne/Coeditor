from coeditor.common import *
from coeditor.retrieval_model import RetrievalEditorModel
from coeditor.api import (
    EditPredictionService,
    QueryRefEditEncoder,
    BatchArgs,
    DecodingArgs,
)

from jsonrpcserver import Success, method, serve, InvalidParams, Result, Error


def start_server(device, port: int = 5042):
    model_path = get_model_dir(True) / "coeditor-large-request-stub"
    model = RetrievalEditorModel.load(model_path)
    model.to(device)
    print(f"Model '{model_path.name}' loaded on device:", device)
    batch_args = BatchArgs.service_default()
    services = dict[Path, EditPredictionService]()

    @method
    def suggestAndApply(project: str, file: str, line: int):
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
                # dec_args=DecodingArgs(do_sample=False, num_beams=8, length_penalty=0.0),
                dec_args=DecodingArgs(
                    do_sample=True, top_p=0.95, marginalize_samples=20
                ),
            )
            print(f"Service created for project: {target_dir}")
            services[target_dir] = service

        print(f"Suggesting edit for line {line} in {file}")
        path = Path(file)
        if not Path.is_absolute(path):
            path = target_dir / path
        try:
            desc = service.suggest_edit(path, line, apply_edit=True)
            return Success(desc)
        except Exception as e:
            return Error(message=str(e))

    print(f"Starting suggestion server at localhost:{port}")
    serve("localhost", port)


if __name__ == "__main__":
    start_server("cuda:2")
