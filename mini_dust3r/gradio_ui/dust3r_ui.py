try:
    import spaces  # type: ignore # noqa: F401

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

import gradio as gr
import os
import torch
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import tempfile

from mini_dust3r.api import (
    OptimizedResult,
    inferece_dust3r_from_paths,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo

from pillow_heif import register_heif_opener  # noqa

register_heif_opener()  # noqa

if gr.NO_RELOAD:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(DEVICE)


def create_blueprint(image_name_list: list[str], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    if len(image_name_list) > 4:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"{log_path}"),
            ),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Spatial3DView(origin=f"{log_path}"),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{log_path}/camera_{i}/pinhole/",
                                contents=[
                                    "+ $origin/**",
                                ],
                            )
                            for i in range(len(image_name_list))
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint


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

    log_path = Path("world")

    if isinstance(image_name_list, str):
        image_name_list = [image_name_list]

    # Extract the parent directory of the images
    if isinstance(image_name_list, str):
        image_name_list = [image_name_list]

    image_path_list: list[Path] = [Path(image) for image in image_name_list]

    optimized_results: OptimizedResult = inferece_dust3r_from_paths(
        image_dir_or_list=image_path_list,
        model=model,
        device=DEVICE,
        batch_size=1,
    )

    blueprint: rrb.Blueprint = create_blueprint(image_name_list, log_path)
    rr.send_blueprint(blueprint)

    rr.set_time_sequence("sequence", 0)
    log_optimized_result(optimized_results, log_path, jpeg_quality=80)
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
