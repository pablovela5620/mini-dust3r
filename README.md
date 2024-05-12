# Mini-Dust3r
A miniature version of [dust3r](https://github.com/naver/dust3r) only for performing inference.
This makes it much easier to use without needing the training/data/eval code.
<p align="center">
  <img src="media/mini-dust3r.gif" alt="example output" width="720" />
</p>


## Installation
Easily installable via pip
```bash
pip install mini-dust3r
```

or from source using [Pixi](pixi.sh)

``` bash
git clone https://github.com/pablovela5620/mini-dust3r.git
pixi run rerun-demo
```

## Minimal Example
Uses [Rerun](http://rerun.io/) to visualize the outputs

```python
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
```

The output from OptimizedResult looks like the following

```python
@dataclass
class OptimizedResult:
    K_b33: Float32[np.ndarray, "b 3 3"]
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"]
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]]
    depth_hw_list: list[Float32[np.ndarray, "h w"]]
    conf_hw_list: list[Float32[np.ndarray, "h w"]]
    masks_list: Bool[np.ndarray, "h w"]
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh
```
Which consists of
* K_b33 - camera intrinsics of shape (b33)
* world_T_cam_b44 - camera to world transformation matrix of shape b44
     in OpenCV convention X - Right Y - Down Z - Forward (RDF)
* rgb_hw3_list - list of RGB images shape (list[hw3])
* depth_hw_list - list of normalized depth maps shape (list[hw])
* conf_hw_list - list of normalized confidence values (list[hw])
* mask_list - list of masks (list[hw])
* point cloud - as a trimesh pointcloud object
* mesh - as a trimesh mesh object

## References
Full credit goes the Naver for their great work on
* [Dust3r](https://github.com/naver/croco)
* [Croco](https://github.com/naver/croco)
