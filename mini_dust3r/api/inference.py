import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import torch
import trimesh
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from simplecv.camera_parameters import (
    Extrinsics,
    Intrinsics,
    PinholeParameters,
    rescale_intri,
)
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from mini_dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.inference import Dust3rResult, inference
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.rerun_log_utils import create_blueprint
from mini_dust3r.utils.image import (
    ImageDict,
    load_images_from_dir_or_list,
    preprocess_rgb,
)
from mini_dust3r.viz import cat_meshes, pts3d_to_trimesh


@dataclass
class OptimizedResult:
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]]
    pinhole_param_list: list[PinholeParameters]
    depth_hw_list: list[Float32[np.ndarray, "h w"]]
    conf_hw_list: list[Float32[np.ndarray, "h w"]]
    masks_list: list[Bool[np.ndarray, "h w"]]
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh

    def rescale_to_size(self, *, height: int, width: int) -> "OptimizedResult":
        """Rescale camera parameters and images to a new size.

        Args:
            new_height: Target height
            new_width: Target width

        Returns:
            A new OptimizedResult with rescaled parameters and images
        """
        # Create a copy to avoid modifying the original
        result = copy.deepcopy(self)

        # Rescale images, depth maps, and confidence maps
        for i in range(len(result.rgb_hw3_list)):
            # Resize RGB images
            result.rgb_hw3_list[i] = cv2.resize(
                result.rgb_hw3_list[i],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Resize depth maps
            result.depth_hw_list[i] = cv2.resize(
                result.depth_hw_list[i],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Resize confidence maps
            result.conf_hw_list[i] = cv2.resize(
                result.conf_hw_list[i],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Resize masks
            result.masks_list[i] = cv2.resize(
                result.masks_list[i].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Rescale pinhole parameters using rescale_intri
        for i, param in enumerate(result.pinhole_param_list):
            result.pinhole_param_list[i].intrinsics = rescale_intri(
                param.intrinsics, target_height=height, target_width=width
            )

        return result


def log_optimized_result(
    optimized_result: OptimizedResult,
    parent_log_path: Path,
    jpeg_quality: int = 95,
    log_depth: bool = True,
    geometry_type: Literal["mesh", "pointcloud", "both"] = "pointcloud",
) -> None:
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    if geometry_type in ["pointcloud", "both"]:
        rr.log(
            f"{parent_log_path}/pointcloud",
            rr.Points3D(
                positions=optimized_result.point_cloud.vertices,
                colors=optimized_result.point_cloud.colors,
            ),
            static=True,
        )

    if geometry_type in ["mesh", "both"]:
        mesh = optimized_result.mesh
        rr.log(
            f"{parent_log_path}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                vertex_colors=mesh.visual.vertex_colors,
                triangle_indices=mesh.faces,
            ),
            static=True,
        )

    pbar = tqdm(
        zip(
            optimized_result.rgb_hw3_list,
            optimized_result.depth_hw_list,
            optimized_result.conf_hw_list,
            optimized_result.pinhole_param_list,
            strict=True,
        ),
        total=len(optimized_result.rgb_hw3_list),
    )
    for rgb_hw3, depth_hw, conf_hw, pinhole_param in pbar:
        camera_log_path = parent_log_path / pinhole_param.name
        # convert image to UInt8 to allow for compression
        rgb_hw3 = (rgb_hw3 * 255).astype(np.uint8)
        log_pinhole(
            camera=pinhole_param,
            cam_log_path=camera_log_path,
            image_plane_distance=0.01,
        )
        rr.log(
            f"{camera_log_path}/pinhole/rgb",
            rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality),
        )
        # use depthimage to log confidence to allow for colormap
        rr.log(
            f"{camera_log_path}/pinhole/confidence",
            rr.DepthImage(conf_hw, colormap=rr.components.Colormap.Turbo),
        )
        if log_depth:
            rr.log(
                f"{camera_log_path}/pinhole/depth",
                rr.DepthImage(depth_hw, draw_order=-5.0),
            )


def to_pinhole_params(
    K_b33: Float32[np.ndarray, "b 3 3"],
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"],
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]],
) -> list[PinholeParameters]:
    pinhole_param_list: list[PinholeParameters] = []
    for idx, (K_33, world_T_cam_44, rgb_hw3) in enumerate(
        zip(K_b33, world_T_cam_b44, rgb_hw3_list, strict=False)
    ):
        h, w, _ = rgb_hw3.shape
        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(K_33[0, 0]),
            fl_y=float(K_33[1, 1]),
            cx=float(K_33[0, 2]),
            cy=float(K_33[1, 2]),
            width=w,
            height=h,
        )
        extri = Extrinsics(
            world_R_cam=world_T_cam_44[:3, :3],
            world_t_cam=world_T_cam_44[:3, 3],
        )
        pinhole_param_list.append(
            PinholeParameters(name=f"camera_{idx}", intrinsics=intri, extrinsics=extri)
        )

    return pinhole_param_list


def scene_to_results(scene: BasePCOptimizer, min_conf_thr: float) -> OptimizedResult:
    assert min_conf_thr > 0 and min_conf_thr < 1, (
        "min_conf_thr should be in the range (0, 1)"
    )
    K_b33: Float32[np.ndarray, "b 3 3"] = scene.get_intrinsics().numpy(force=True)
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"] = scene.get_im_poses().numpy(
        force=True
    )
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]] = scene.imgs
    pinhole_params: list[PinholeParameters] = to_pinhole_params(
        K_b33, world_T_cam_b44, rgb_hw3_list
    )

    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth.numpy(force=True) for depth in scene.get_depthmaps()
    ]

    conf_hw_list: list[Float32[np.ndarray, "h w"]] = [
        c.numpy(force=True) for c in scene.im_conf
    ]
    conf_hw_list = [c / c.max() for c in conf_hw_list]

    pts3d_list: list[Float32[np.ndarray, "h w 3"]] = [
        pt3d.numpy(force=True) for pt3d in scene.get_pts3d()
    ]
    masks_list: list[Bool[np.ndarray, "h w"]] = [
        conf_hw > min_conf_thr for conf_hw in conf_hw_list
    ]

    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth_hw * mask
        for depth_hw, mask in zip(depth_hw_list, masks_list, strict=True)
    ]

    point_cloud: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(pts3d_list, masks_list, strict=True)]
    )
    colors: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(rgb_hw3_list, masks_list, strict=True)]
    )
    point_cloud = trimesh.PointCloud(
        point_cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)
    )

    meshes = []
    pbar = tqdm(
        zip(rgb_hw3_list, pts3d_list, masks_list, strict=True), total=len(rgb_hw3_list)
    )
    for rgb_hw3, pts3d, mask in pbar:
        meshes.append(pts3d_to_trimesh(rgb_hw3, pts3d, mask))

    mesh = trimesh.Trimesh(**cat_meshes(meshes))
    optimised_result = OptimizedResult(
        rgb_hw3_list=rgb_hw3_list,
        pinhole_param_list=pinhole_params,
        depth_hw_list=depth_hw_list,
        conf_hw_list=conf_hw_list,
        masks_list=masks_list,
        point_cloud=point_cloud,
        mesh=mesh,
    )
    return optimised_result


def inferece_dust3r_from_rgb(
    rgb_list: list[UInt8[np.ndarray, "H W 3"]],
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    niter: int = 100,
    schedule: Literal["linear", "cosine"] = "linear",
    min_conf_thr: float = 0.25,
) -> OptimizedResult:
    """
    Perform inference using the Dust3r algorithm.

    Args:
        rgb_list list[np.ndarray]: Path to the directory containing images or a list of image paths.
        model (AsymmetricCroCo3DStereo): The Dust3r model to use for inference.
        device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
        batch_size (int, optional): The batch size for inference. Defaults to 1.
        image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
        niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
        schedule (Literal["linear", "cosine"], optional): The learning rate schedule for the global alignment optimization. Defaults to "linear".
        min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 0.5.

    Returns:
        OptimizedResult: The optimized result containing the RGB, depth, and confidence images.

    Raises:
        ValueError: If `image_dir_or_list` is neither a list of paths nor a path.
    """
    assert min_conf_thr > 0 and min_conf_thr < 1, (
        "min_conf_thr should be in the range (0, 1)"
    )
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

    mode: GlobalAlignerMode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene: BasePCOptimizer = global_aligner(
        dust3r_output=output, device=device, mode=mode
    )

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=0.01
        )

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results(scene, min_conf_thr)
    return optimized_result


@dataclass
class InferenceConfig:
    rr_config: RerunTyroConfig
    image_dir: Path
    image_size: Literal[224, 512] = 512
    niter: int = 100
    schedule: Literal["linear", "cosine"] = "linear"
    min_conf_thr: float = 0.25
    return_original_size: bool = True

    def __post_init__(self):
        assert self.min_conf_thr > 0 and self.min_conf_thr < 1, (
            "min_conf_thr should be in the range (0, 1)"
        )


def run_inference(config: InferenceConfig):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    parent_log_path = Path("world")
    model: AsymmetricCroCo3DStereo = AsymmetricCroCo3DStereo.from_pretrained(
        "pablovela5620/dust3r"
    ).to(device)

    image_name_list: list[Path] = [
        image_path for image_path in config.image_dir.iterdir() if image_path.is_file()
    ]
    blueprint = create_blueprint(image_name_list, parent_log_path)
    rr.send_blueprint(blueprint)

    rgb_list: list[UInt8[np.ndarray, "H W 3"]] = load_images_from_dir_or_list(
        config.image_dir
    )
    assert len(rgb_list) > 0, f"No images found in {config.image_dir}"
    optimized_results: OptimizedResult = inferece_dust3r_from_rgb(
        rgb_list=rgb_list,
        model=model,
        device=device,
        batch_size=1,
        image_size=config.image_size,
        niter=config.niter,
        schedule=config.schedule,
        min_conf_thr=config.min_conf_thr,
    )
    original_height, original_width, _ = rgb_list[0].shape
    resized_results = optimized_results.rescale_to_size(
        height=original_height, width=original_width
    )
    log_optimized_result(resized_results, parent_log_path)
