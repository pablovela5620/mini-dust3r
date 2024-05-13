import gradio as gr
import torch
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import uuid


from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to(DEVICE)


def predict(image_name_list: list[str]):
    uuid_str = str(uuid.uuid4())
    filename = Path(f"/tmp/gradio/{uuid_str}.rrd")
    rr.init(f"{uuid_str}")
    log_path = Path("world")
    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_name_list,
        model=model,
        device=DEVICE,
        batch_size=1,
    )
    rr.set_time_sequence("sequence", 0)
    log_optimized_result(optimized_results, log_path)
    # blueprint = rrb.Spatial3DView(origin="cube")
    rr.save(filename.as_posix())
    return filename.as_posix()


with gr.Blocks(
    css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
    title="Mini-DUSt3R Demo",
) as demo:
    # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
    gr.HTML('<h2 style="text-align: center;">Mini-DUSt3R Demo</h2>')
    with gr.Column():
        inputfiles = gr.File(file_count="multiple")
        rerun_viewer = Rerun(height=900)

        run_btn = gr.Button("Run")
        run_btn.click(fn=predict, inputs=[inputfiles], outputs=[rerun_viewer])

demo.launch()
