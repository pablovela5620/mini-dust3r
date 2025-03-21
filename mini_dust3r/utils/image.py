# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from einops import rearrange
from jaxtyping import Float32, Int32
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

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


def load_images_from_dir_or_list(
    image_dir_or_list: list[Path] | Path,
) -> list[np.ndarray]:
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    supported_images_extensions: list[str] = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)
    # get a list of paths of all images in supported_images_extensions
    if isinstance(image_dir_or_list, list):
        paths_list: list[Path] = image_dir_or_list
    else:
        paths_list: list[Path] = list(image_dir_or_list.iterdir())

    # filter paths_list
    paths_list = (
        path
        for path in paths_list
        if path.is_file() and path.suffix.lower() in supported_images_extensions
    )
    return [read_image_path(path) for path in paths_list]


def load_and_preprocess_images(
    image_dir_or_list: list[Path] | Path,
    size: Literal[224, 512],
    square_ok: bool = False,
) -> list[ImageDict]:
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    rgb_list: list[np.ndarray] = load_images_from_dir_or_list(image_dir_or_list)

    imgs: list[ImageDict] = []
    for idx, rgb_img in enumerate(tqdm(rgb_list, desc="Processing images")):
        # Read the image and get its original dimensions.
        preprocessed_img: Float32[torch.Tensor, "3 H W"] = preprocess_rgb(
            rgb_img, size, square_ok
        )
        imgs.append(
            ImageDict(
                img=rearrange(preprocessed_img, "c h w -> 1 c h w"),
                true_shape=np.int32(
                    [[preprocessed_img.shape[1], preprocessed_img.shape[2]]]
                ),
                idx=idx,
                instance=str(idx),
            )
        )

    assert imgs, f"no images found in {image_dir_or_list}"
    return imgs


def preprocess_rgb(
    rgb_img: np.ndarray, size: Literal[224, 512], square_ok: bool = False
) -> Float32[torch.Tensor, "3 H W"]:
    """Preprocess a single image for DUSt3R.

    Args:
        rgb_img: Image in RGB format
        size: Target size for the longer edge
        square_ok: If False and the image is square (with size 512), a 3:4 crop is applied

    Returns:
        ImageDict: Preprocessed image
    """
    resized_img = resize_and_crop_cv2(rgb_img, size, square_ok)
    img_tensor = ImgNorm(resized_img)
    return img_tensor


def read_image_path(image_path: Path) -> np.ndarray:
    """Read an image file to a numpy array in RGB format.

    Args:
        image_path: Path to the image file

    Returns:
        Numpy array containing the image in RGB format
    """
    assert image_path.exists(), f"Image path does not exist: {image_path}"
    img_pil = PIL.Image.open(image_path)
    img_pil = exif_transpose(img_pil).convert("RGB")
    return np.array(img_pil)


def resize_and_crop_cv2(
    rgb_img: np.ndarray, size: Literal[224, 512], square_ok: bool = False
) -> np.ndarray:
    """
    Resize and crop an image from the given path.

    The image is first resized based on the target size:
      - For size 224, the scaling factor is determined by the maximum ratio of the original width or height to 224.
      - For size 512, the image is scaled so that its longest edge becomes 512.

    After resizing, the image is center-cropped:
      - For size 224, a square crop is performed.
      - For size 512, if square_ok is False and the image is square, the crop uses a 3:4 aspect ratio.

    Parameters:
        image_path (Path): Path to the image file.
        size (Literal[224, 512]): Target size for the longer edge.
        square_ok (bool): If False and the image is square (with size 512), a 3:4 crop is applied.

    Returns:
        np.ndarray: The resized and cropped image.
    """
    original_height, original_width = rgb_img.shape[:2]

    # Determine the scaling factor.
    if size == 224:
        size_ratio = max(original_width / size, original_height / size)
        target_long_edge = round(size * size_ratio)
        scale = target_long_edge / max(original_width, original_height)
    elif size == 512:
        scale = size / max(original_width, original_height)
    else:
        raise ValueError("size must be either 224 or 512")

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    interpolation = cv2.INTER_LANCZOS4 if scale < 1 else cv2.INTER_CUBIC
    resized_img = cv2.resize(
        rgb_img, (new_width, new_height), interpolation=interpolation
    )

    # Center crop the resized image.
    height, width = resized_img.shape[:2]
    center_x, center_y = width // 2, height // 2

    if size == 224:
        half_crop = min(center_x, center_y)
        cropped_img = resized_img[
            center_y - half_crop : center_y + half_crop,
            center_x - half_crop : center_x + half_crop,
        ]
    else:
        half_width = ((2 * center_x) // 16) * 8
        half_height = ((2 * center_y) // 16) * 8
        if not square_ok and width == height:
            half_height = int(3 * half_width / 4)
        cropped_img = resized_img[
            center_y - half_height : center_y + half_height,
            center_x - half_width : center_x + half_width,
        ]
    return cropped_img
