# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
from PIL.ImageFile import ImageFile
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf

from typing import Literal, TypedDict
from jaxtyping import Float32, Int32
from pathlib import Path
import cv2

try:
    from pillow_heif import register_heif_opener  # noqa

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ImageDict(TypedDict):
    img: Float32[torch.Tensor, "b c h w"]
    true_shape: tuple[int, int] | Int32[torch.Tensor, "b 2"]
    idx: int | list[int]
    instance: str | list[str]


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    max_edge_size = max(img.size)
    if max_edge_size > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif max_edge_size <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / max_edge_size)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(
    image_dir_or_list: list[Path] | Path,
    size: Literal[224, 512],
    square_ok: bool = False,
    verbose: bool = True,
) -> list[ImageDict]:
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    supported_images_extensions: list[str] = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)
    # get a list of paths of all images in supported_images_extensions
    if isinstance(image_dir_or_list, list):
        path_iter: list[Path] = image_dir_or_list
    else:
        path_iter: list[Path] = list(image_dir_or_list.iterdir())

    imgs: list[ImageDict] = []
    for path in path_iter:
        if not path.is_file() or path.suffix.lower() not in supported_images_extensions:
            continue
        img: PIL.Image.Image = resize_and_crop_pil(
            image_path=path, size=size, square_ok=square_ok
        )
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
        # img = resize_and_crop_cv2(image_path=path, size=size, square_ok=square_ok)
        # H, W = img.shape[:2]
        # imgs.append(
        #     dict(
        #         img=ImgNorm(img)[None],
        #         true_shape=np.int32([[H, W]]),
        #         idx=len(imgs),
        #         instance=str(len(imgs)),
        #     )
        # )

    assert imgs, f"no images found in {image_dir_or_list}"
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs


def resize_and_crop_pil(
    image_path: Path, size: Literal[224, 512], square_ok: bool = False
) -> PIL.Image.Image:
    assert image_path.exists(), f"bad {image_path=}"
    img_pil_path: ImageFile = PIL.Image.open(image_path)
    img: PIL.Image.Image = exif_transpose(img_pil_path).convert("RGB")
    original_w, original_h = img.size
    ## First Resize
    if size == 224:
        # resize short side to 224 (then crop)
        size_ratio: float = max(original_w / size, original_h / size)
        long_edge_size: int = round(size * size_ratio)
        img = _resize_pil_image(img, long_edge_size)
    elif size == 512:
        # resize long side to 512
        img = _resize_pil_image(img, size)

    ## Then Crop
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half: int = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    resized_w, resized_h = img.size
    print(
        f" - adding {image_path} with resolution {original_w}x{original_h} --> {resized_w}x{resized_h}"
    )
    return img


def resize_and_crop_cv2(
    image_path: Path, size: Literal[224, 512], square_ok: bool = False
) -> np.ndarray:
    assert image_path.exists(), f"bad {image_path=}"
    img_bgr = cv2.imread(str(image_path))
    assert img_bgr is not None, f"Failed to load {image_path}"
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    original_h, original_w = img.shape[:2]

    # First Resize
    if size == 224:
        size_ratio = max(original_w / size, original_h / size)
        long_edge_size = round(size * size_ratio)
        scale = long_edge_size / max(original_w, original_h)
    elif size == 512:
        scale = size / max(original_w, original_h)

    new_w, new_h = int(round(original_w * scale)), int(round(original_h * scale))
    interp = cv2.INTER_LANCZOS4 if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Then Crop
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2

    if size == 224:
        half = min(cx, cy)
        img = img[cy - half : cy + half, cx - half : cx + half]
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = int(3 * halfw / 4)
        img = img[cy - halfh : cy + halfh, cx - halfw : cx + halfw]

    resized_h, resized_w = img.shape[:2]
    print(
        f" - adding {image_path} with resolution {original_w}x{original_h} --> {resized_w}x{resized_h}"
    )
    return img
