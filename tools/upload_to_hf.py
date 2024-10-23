from argparse import ArgumentParser, Namespace
from typing import TypedDict
import os
from huggingface_hub import upload_file, upload_folder
from tqdm import tqdm
from pathlib import Path


class FileUpload(TypedDict):
    local_path: str
    repo_path: str


def main(upload_examples: bool) -> None:
    space_id: str | None = os.environ.get("SPACE_ID")
    whl_path: str | None = os.environ.get("WHL_PATH")

    assert space_id is not None, "Please set the SPACE_ID environment variable"
    assert whl_path is not None, "Please set the WHL_PATH environment variable"

    files_to_upload: list[FileUpload] = [
        {"local_path": "./tools/app.py", "repo_path": "app.py"},
        {"local_path": "./tools/gradio_app.py", "repo_path": "gradio_app.py"},
        {
            "local_path": f"./dist/{whl_path}",
            "repo_path": f"./dist/{whl_path}",
        },
        {"local_path": "./pyproject.toml", "repo_path": "pyproject.toml"},
        {"local_path": "./pixi.lock", "repo_path": "pixi.lock"},
        {"local_path": "./.pixi.sh", "repo_path": ".pixi.sh"},
    ]

    for file in tqdm(files_to_upload, desc="Uploading files"):
        assert os.path.exists(
            file["local_path"]
        ), f"File {file['local_path']} does not exist"
        with open(file["local_path"], "rb") as fobj:
            upload_file(
                path_or_fileobj=fobj,
                path_in_repo=file["repo_path"],
                repo_id=space_id,
                repo_type="space",
            )

    if upload_examples:
        examples_path = Path("examples")
        assert examples_path.exists(), f"Examples folder {examples_path} does not exist"
        upload_folder(
            folder_path=str(examples_path),
            path_in_repo=str(examples_path),
            repo_id=space_id,
            repo_type="space",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--upload-examples", action="store_true")
    args: Namespace = parser.parse_args()
    main(upload_examples=args.upload_examples)
