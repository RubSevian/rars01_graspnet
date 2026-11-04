"""Grasp NMS and pose conversion; no neural-network/model imports.

Shared by the inference worker, the executable planner and GUI.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from .graspnet_paths import prepare_graspnet_api_imports

prepare_graspnet_api_imports()
from graspnetAPI import Grasp, GraspGroup
from .transforms import (
    graspnet_rotation_to_rars_tcp_rotation,
    graspnet_rotation_to_tcp_rotation,
    transform_graspnet_grasp_to_end_link_base_with_retreat,
    transform_grasp_pose_to_base_with_retreat,
)


def nms_grasp_group(
    grasps: GraspGroup,
    translation_thresh_m: float = 0.03,
    rotation_thresh_rad: float = np.deg2rad(30.0),
) -> GraspGroup:
    """Run native GraspNet NMS, with an equivalent CPU fallback on Jetson.

    The upstream extension is optional and often absent from ARM64 builds.  The
    fallback keeps the highest-score candidate in each translation/orientation
    neighbourhood, so top-K selection never degenerates into near-duplicates.
    """
    if len(grasps) < 2:
        return GraspGroup(grasps.grasp_group_array.copy())
    try:
        return grasps.nms(translation_thresh_m, rotation_thresh_rad)
    except Exception as exc:
        ranked = GraspGroup(grasps.grasp_group_array.copy())
        ranked.sort_by_score()
        keep: list[int] = []
        for index, candidate in enumerate(ranked):
            duplicate = False
            for kept_index in keep:
                selected = ranked[kept_index]
                translation_delta = float(
                    np.linalg.norm(candidate.translation - selected.translation)
                )
                relative = candidate.rotation_matrix.T @ selected.rotation_matrix
                cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
                angle = float(np.arccos(cosine))
                if translation_delta < translation_thresh_m and angle < rotation_thresh_rad:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(index)
        print(f"[G] Native NMS unavailable ({exc}); using CPU NMS: {len(grasps)} -> {len(keep)}")
        return ranked[np.asarray(keep, dtype=np.int64)]


def grasp_to_base_poses(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
    tcp_convention: str = "rars01",
    T_grasp_tcp: Optional[np.ndarray] = None,
    allow_parallel_flip: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if tcp_convention == "rars01" and T_grasp_tcp is not None:
        if abs(float(insertion_depth_m)) > 1e-12:
            raise ValueError(
                "RARS01 End_link targets use GraspNet depth directly; "
                "insertion_depth_m must be 0"
            )
        return transform_graspnet_grasp_to_end_link_base_with_retreat(
            np.asarray(grasp.translation, dtype=np.float64),
            np.asarray(grasp.rotation_matrix, dtype=np.float64),
            float(grasp.depth),
            T_cam2base,
            T_grasp_tcp,
            pregrasp_offset_m,
            retreat_offset_m,
            allow_parallel_flip,
        )
    rotation_fn = (
        graspnet_rotation_to_rars_tcp_rotation
        if tcp_convention == "rars01"
        else graspnet_rotation_to_tcp_rotation
    )
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        rotation_fn(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )
