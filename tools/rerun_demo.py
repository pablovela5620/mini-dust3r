import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir=image_dir,
        model=model,
        device=device,
        batch_size=1,
    )
    log_optimized_result(optimized_results, Path("world"))


if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    main(args.image_dir)
    rr.script_teardown(args)
