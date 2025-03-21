try:
    import spaces  # type: ignore # noqa: F401

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gradio_rerun import Rerun
from jaxtyping import UInt8
from pillow_heif import register_heif_opener  # noqa

from mini_dust3r.api import (
    OptimizedResult,
    inferece_dust3r_from_rgb,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.rerun_log_utils import create_blueprint
from mini_dust3r.utils.image import load_images_from_dir_or_list

register_heif_opener()  # noqa

if gr.NO_RELOAD:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained("pablovela5620/dust3r").to(DEVICE)


@rr.thread_local_stream("dust3r_rrd")
def predict(
    image_name_list: list[str] | str,
    pending_cleanup: list[str] = [],
    progress=gr.Progress(track_tqdm=True),
) -> str:
    # check if is list or string and if not raise error
    if not isinstance(image_name_list, list) and not isinstance(image_name_list, str):
        raise gr.Error(
            f"Input must be a list of strings or a string, got: {type(image_name_list)}"
        )

    temp = tempfile.NamedTemporaryFile(prefix="dust3r_", suffix=".rrd", delete=False)
    pending_cleanup.append(temp.name)

    parent_log_path = Path("world")

    if isinstance(image_name_list, str):
        image_name_list = [image_name_list]

    # Extract the parent directory of the images
    if isinstance(image_name_list, str):
        image_name_list = [image_name_list]

    image_path_list: list[Path] = [Path(image) for image in image_name_list]
    rgb_list: list[UInt8[np.ndarray, "H W 3"]] = load_images_from_dir_or_list(
        image_path_list
    )

    optimized_results: OptimizedResult = inferece_dust3r_from_rgb(
        rgb_list=rgb_list,
        model=model,
        device=DEVICE,
        batch_size=1,
    )

    blueprint: rrb.Blueprint = create_blueprint(image_path_list, parent_log_path)
    rr.send_blueprint(blueprint)

    rr.set_time_sequence("sequence", 0)
    log_optimized_result(optimized_results, parent_log_path, jpeg_quality=80)
    rr.save(temp.name, default_blueprint=blueprint)
    return temp.name


if IN_SPACES:
    predict = spaces.GPU(predict)


def cleaup_rrds(pending_cleanup: list) -> None:
    for f in pending_cleanup:
        os.unlink(f)


def get_multi_example_files(
    example_parent_dir: Path, patterns: list[str]
) -> list[list[list[str]]]:
    final_example_files: list[list[str]] = []
    example_multi_dirs: list[Path] = sorted(
        directory for directory in example_parent_dir.glob("*")
    )
    for directory in example_multi_dirs:
        example_multi_files: list[str] = sorted(
            str(file) for pattern in patterns for file in directory.glob(pattern)
        )
        final_example_files.append([example_multi_files])

    return final_example_files


with gr.Blocks() as mini_dust3r_ui:
    pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleaup_rrds)
    with gr.Tab(label="Single Image"):
        with gr.Column():
            single_image = gr.Image(type="filepath", height=300)
            run_btn_single = gr.Button("Run")
            rerun_viewer_single = Rerun(height=900)
            run_btn_single.click(
                fn=predict,
                inputs=[single_image, pending_cleanup],
                outputs=[rerun_viewer_single],
            )

            example_single_dir = Path("examples/single_image")
            patterns: list[str] = ["*.jpg", "*.png", "*.jpeg", "*.HEIC"]
            example_single_files = sorted(
                file
                for pattern in patterns
                for file in example_single_dir.glob(pattern)
            )

            examples_single = gr.Examples(
                examples=example_single_files,
                inputs=[single_image],
                outputs=[rerun_viewer_single],
                fn=predict,
                cache_examples="lazy",
            )
    with gr.Tab(label="Multi Image"):
        with gr.Column():
            multi_files = gr.File(file_count="multiple")
            run_btn_multi = gr.Button("Run")
            rerun_viewer_multi = Rerun(height=900)
            run_btn_multi.click(
                fn=predict,
                inputs=[multi_files, pending_cleanup],
                outputs=[rerun_viewer_multi],
            )

            example_multi_dir = Path("examples/multi_image")
            # get all directories in examples/multi_image
            examples_multi_files: list[list[list[str]]] = get_multi_example_files(
                example_multi_dir, patterns
            )

            examples_multi = gr.Examples(
                examples=examples_multi_files,
                inputs=[multi_files],
                outputs=[rerun_viewer_multi],
                fn=predict,
                cache_examples="lazy",
            )
