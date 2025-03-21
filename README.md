# Mini-Dust3r
A miniature version of [dust3r](https://github.com/naver/dust3r) only for performing inference.
This makes it much easier to use without needing the training/data/eval code. Tested on Linux, Apple Silicon Macs, and Windows (Thanks @Vincentqyw)
<p align="center">
  <img src="media/mini-dust3r.gif" alt="example output" width="720" />
</p>


## Installation
Easily installable via pip
```bash
pip install mini-dust3r
```

## Demo
A hosted demo can be found on huggingface here <a href='https://huggingface.co/spaces/pablovela5620/mini-dust3r'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

or from source using [Pixi](http://pixi.sh)

``` bash
git clone https://github.com/pablovela5620/mini-dust3r.git
pixi run app
```

You can also just use rerun demo directly with
```bash
pixi run rerun-demo
```

## Minimal Example
Uses [Rerun](http://rerun.io/) to visualize the outputs

```python
import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import OptimizedResult, inferece_dust3r_from_rgb, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images_from_dir_or_list


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    # Load images from directory
    rgb_list:list[UInt8[np.ndarray, "H W 3"]]  = load_images_from_dir_or_list(image_dir)

    optimized_results: OptimizedResult = inferece_dust3r_from_rgb(
        rgb_list=rgb_list,
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

## Calling Model Directly
Requires converting rgb numpy arrays to torch tensors, making a dict that is defined in typed_dict ImageDict
and generating pairs to be fed into the Dust3r model.
```python
    processed_imgs: list[Float32[torch.Tensor, "3 H W"]] = [
        preprocess_rgb(rgb_img, image_size, square_ok=False) for rgb_img in rgb_list
    ]
    imgs: list[ImageDict] = [
        ImageDict(
            img=rearrange(img, "c h w -> 1 c h w"),
            true_shape=np.int32([[img.shape[1], img.shape[2]]]),
            idx=idx,
            instance=str(idx),
        )
        for idx, img in enumerate(processed_imgs)
    ]
    assert imgs, "no images found"

    # if only one image was loaded, duplicate it to feed into stereo network
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(
        imgs, scene_graph="complete", prefilter=None, symmetrize=True
    )
    output: Dust3rResult = inference(pairs, model, device, batch_size=batch_size)
```

## Inputs and Outputs

### Inference Function

```python
def inferece_dust3r_from_rgb(
    rgb_list: list[np.ndarray],
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    niter: int = 100,
    schedule: Literal["linear", "cosine"] = "linear",
    min_conf_thr: float = 0.25,
) -> OptimizedResult:
```
Consists of
* rgb_list - List of RGB images as numpy arrays
* model - The Dust3r model to use for inference
* device - device to use for inference ("cpu", "cuda", or "mps")
* batch_size - The batch size for inference. Defaults to 1.
* image_size - The size of the input images. Defaults to 512.
* niter - The number of iterations for the global alignment optimization. Defaults to 100.
* schedule - The learning rate schedule for the global alignment optimization. Defaults to "linear"
* min_conf_thr - The minimum confidence threshold for the optimized result. Defaults to 0.25.

### Output from OptimizedResult

```python
@dataclass
class OptimizedResult:
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]]
    pinhole_param_list: list[PinholeParameters]
    depth_hw_list: list[Float32[np.ndarray, "h w"]]
    conf_hw_list: list[Float32[np.ndarray, "h w"]]
    masks_list: list[Bool[np.ndarray, "h w"]]
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh
```
Consists of
* rgb_hw3_list - list of RGB images shape (list[hw3])
* pinhole_param_list - list of camera parameters (intrinsics and extrinsics)
* depth_hw_list - list of normalized depth maps shape (list[hw])
* conf_hw_list - list of normalized confidence values (list[hw])
* masks_list - list of masks (list[hw])
* point_cloud - as a trimesh pointcloud object
* mesh - as a trimesh mesh object

## References
Full credit goes the Naver for their great work on
* [Dust3r](https://github.com/naver/dust3r)
* [Croco](https://github.com/naver/croco)
