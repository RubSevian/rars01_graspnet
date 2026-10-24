"""Executable candidate selection shared with the real GUI; no SDK ownership.

Extracted without changing ranking, transforms or Cartesian IK checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from utils import graspnet_utils
from graspnetAPI import Grasp, GraspGroup
from rars01_graspnet.real_grasp_ik import IkChecker, IkSolution, _plan_cartesian_segment
from rars01_graspnet.grasp_selection import normalized_grasp_costs, joint_limit_cost, motion_cost, best_validated_candidate
_PARALLEL_FLIP_X = np.diag([1., -1., -1.])

@dataclass(frozen=True)
class ExecutableGrasp:
    grasp: Grasp
    grasp6d: tuple[float, ...]
    pregrasp6d: tuple[float, ...]
    retreat6d: tuple[float, ...]
    pregrasp_joints: np.ndarray
    grasp_joints: np.ndarray
    retreat_joints: np.ndarray
    approach_joint_waypoints: tuple[np.ndarray, ...] = ()
    retreat_joint_waypoints: tuple[np.ndarray, ...] = ()
    flip: bool = False
    raw_score: float = 0.0
    normalized_score: float = 0.0
    cost: float = float("inf")


def _rank_grasps(grasps: GraspGroup, apply_nms: bool = True) -> GraspGroup:
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    if apply_nms and len(ranked) > 1:
        ranked = graspnet_utils.nms_grasp_group(ranked)
    ranked.sort_by_score()
    return ranked


def _parallel_flip_grasp(grasp: Grasp) -> Grasp:
    """Return the same parallel-jaw grasp with the transverse axes reversed."""
    rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64) @ _PARALLEL_FLIP_X
    return Grasp(
        float(grasp.score),
        float(grasp.width),
        float(grasp.height),
        float(grasp.depth),
        rotation,
        np.asarray(grasp.translation, dtype=np.float64),
        int(grasp.object_id),
    )


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    return float(pose6d[2]) >= float(min_z)


def _translate_pose(
    pose6d: tuple[float, ...], offset_base_m: np.ndarray
) -> tuple[float, ...]:
    pose = np.asarray(pose6d, dtype=np.float64).copy()
    pose[:3] += np.asarray(offset_base_m, dtype=np.float64).reshape(3)
    return tuple(float(value) for value in pose)


def _select_executable_grasp(
    ik_checker: IkChecker,
    grasp_driver: GraspDriver,
    grasps: GraspGroup,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    max_grasp_depth_m: float,
    min_base_z_m: float,
    min_jaw_z_m: float,
    robot_backend: str,
    allow_parallel_flip: bool,
    apply_nms: bool,
    position_compensation_base_m: np.ndarray,
    candidate_limit: int,
    cartesian_ik_cfg: dict[str, Any],
    candidate_selection_cfg: dict[str, Any],
    path_validator=None,
    rejection_diagnostics=None,
    local_height_filters_enabled: bool = True,
    cancel_check=None,
    selection_metrics=None,
) -> Optional[ExecutableGrasp]:
    if type(local_height_filters_enabled) is not bool:
        raise ValueError("local_height_filters_enabled must be boolean")
    def reject(reason, **parameters):
        if rejection_diagnostics is not None:
            rejection_diagnostics.append(dict(reason=reason, **parameters))
    ranked = _rank_grasps(grasps, apply_nms=apply_nms)
    top_count = min(len(ranked), candidate_limit)
    if top_count == 0:
        reject("NO_CANDIDATES")
        return None
    top_scores = np.asarray(ranked.scores[:top_count], dtype=np.float64)
    grasp_costs = normalized_grasp_costs(top_scores)
    weight_grasp = float(candidate_selection_cfg["weight_grasp"])
    weight_joint = float(candidate_selection_cfg["weight_joint"])
    weight_motion = float(candidate_selection_cfg["weight_motion"])
    motion_normalization = float(candidate_selection_cfg["motion_normalization"])
    skipped_low = 0
    skipped_jaw = 0
    skipped_depth = 0
    skipped_ik = 0
    worst_err = 0.0

    pose_candidates = []
    for idx in range(top_count):
        grasp = ranked[idx]
        if float(grasp.depth) > max_grasp_depth_m:
            skipped_depth += 1
            reject("DEPTH_LIMIT", rank=idx+1, measured_m=float(grasp.depth), limit_m=max_grasp_depth_m)
            print(f"[Candidate] rank={idx + 1} reject=depth")
            continue
        orientation_candidates = (
            (grasp, _parallel_flip_grasp(grasp)) if allow_parallel_flip else (grasp,)
        )
        for branch_index, oriented_grasp in enumerate(orientation_candidates):
            T_grasp_tcp = grasp_driver.grasp_tcp_transform(float(oriented_grasp.width))
            grasp6d, pre6d, retreat6d = graspnet_utils.grasp_to_base_poses(
                oriented_grasp,
                T_cam2base,
                pregrasp_offset_m,
                retreat_offset_m,
                insertion_depth_m,
                tcp_convention=robot_backend,
                T_grasp_tcp=T_grasp_tcp,
                # Both equivalent parallel-gripper orientations are checked here.
                allow_parallel_flip=False,
            )
            print(
                "[G/transform] "
                f"translation={np.round(oriented_grasp.translation, 6).tolist()} "
                f"depth={float(oriented_grasp.depth):.6f} "
                f"approach_axis_cam={np.round(oriented_grasp.rotation_matrix[:, 0], 6).tolist()} "
                f"End_link_target_base={np.round(np.asarray(grasp6d[:3]), 6).tolist()}"
            )
            grasp6d = _translate_pose(grasp6d, position_compensation_base_m)
            pre6d = _translate_pose(pre6d, position_compensation_base_m)
            retreat6d = _translate_pose(retreat6d, position_compensation_base_m)
            pose_candidates.append(
                (
                    idx,
                    branch_index,
                    oriented_grasp,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    float(grasp.score),
                    float(1.0 - grasp_costs[idx]),
                    float(grasp_costs[idx]),
                )
            )

    current_joints = ik_checker.current_joints()
    lower, upper = ik_checker.joint_limits()
    executable: list[ExecutableGrasp] = []
    for candidate in pose_candidates:
        if cancel_check is not None:cancel_check()
        (
            original_idx,
            branch_index,
            grasp,
            grasp6d,
            pre6d,
            retreat6d,
            raw_score,
            normalized_score,
            grasp_cost,
        ) = candidate
        flip = bool(branch_index)
        candidate_label = f"rank={original_idx + 1} flip={int(flip)}"
        if local_height_filters_enabled and not all(
            _pose_z_ok(pose, min_base_z_m) for pose in (pre6d, grasp6d, retreat6d)
        ):
            skipped_low += 1
            reject("BASE_HEIGHT_LIMIT", candidate=candidate_label, minimum_m=min(float(p[2]) for p in (pre6d,grasp6d,retreat6d)), limit_m=min_base_z_m)
            print(f"[Candidate] {candidate_label} reject=base_z")
            continue
        if (
            local_height_filters_enabled
            and robot_backend == "rars01"
            and min(
                grasp_driver.minimum_jaw_height(pose, float(grasp.width))
                for pose in (pre6d, grasp6d, retreat6d)
            ) < min_jaw_z_m
        ):
            skipped_jaw += 1
            reject("JAW_HEIGHT_LIMIT", candidate=candidate_label, limit_m=min_jaw_z_m)
            print(f"[Candidate] {candidate_label} reject=jaw_z")
            continue

        if hasattr(ik_checker, "context"):
            ik_checker.context = dict(candidate=candidate_label, stage="pregrasp", waypoint=0)
        pre = ik_checker.solve(*pre6d, reference_joints=current_joints)
        approach_waypoints: tuple[np.ndarray, ...] = ()
        retreat_waypoints: tuple[np.ndarray, ...] = ()
        if robot_backend == "rars01" and pre.success:
            approach_waypoints = _plan_cartesian_segment(
                ik_checker, pre6d, grasp6d, pre.joints, cartesian_ik_cfg, "approach"
            ) or ()
            if approach_waypoints:
                grasp_solution = IkSolution(True, 0.0, approach_waypoints[-1])
                retreat_waypoints = _plan_cartesian_segment(
                    ik_checker,
                    grasp6d,
                    retreat6d,
                    grasp_solution.joints,
                    cartesian_ik_cfg,
                    "retreat",
                ) or ()
                retreat = (
                    IkSolution(True, 0.0, retreat_waypoints[-1])
                    if retreat_waypoints
                    else IkSolution(False, float("inf"), grasp_solution.joints)
                )
            else:
                grasp_solution = IkSolution(False, float("inf"), pre.joints)
                retreat = IkSolution(False, float("inf"), pre.joints)
        else:
            grasp_solution = (
                ik_checker.solve(*grasp6d, reference_joints=pre.joints)
                if pre.success else IkSolution(False, pre.error, pre.joints)
            )
            retreat = (
                ik_checker.solve(*retreat6d, reference_joints=grasp_solution.joints)
                if grasp_solution.success
                else IkSolution(False, grasp_solution.error, grasp_solution.joints)
            )
        worst_err = max(worst_err, pre.error, grasp_solution.error, retreat.error)
        if not (pre.success and grasp_solution.success and retreat.success):
            skipped_ik += 1
            reject("IK_OR_CARTESIAN_PATH", candidate=candidate_label,
                   failed_stage="pregrasp" if not pre.success else "approach" if not grasp_solution.success else "retreat")
            print(f"[Candidate] {candidate_label} reject=IK_or_cartesian_path")
            continue

        # This is the exact chain that will be streamed to the arm.
        chain = np.vstack(
            (
                current_joints,
                pre.joints,
                *(approach_waypoints or (grasp_solution.joints,)),
                *(retreat_waypoints or (retreat.joints,)),
            )
        )
        limit_cost = joint_limit_cost(chain, lower, upper)
        path_cost = motion_cost(chain, lower, upper, motion_normalization)
        total_cost = (
            weight_grasp * grasp_cost
            + weight_joint * limit_cost
            + weight_motion * path_cost
        )
        print(
            f"[Candidate] {candidate_label} raw_score={raw_score:.6f} "
            f"normalized_score={normalized_score:.6f} J_grasp={grasp_cost:.6f} "
            f"J_joint={limit_cost:.6f} J_motion={path_cost:.6f} J_total={total_cost:.6f}"
        )
        executable.append(
            ExecutableGrasp(
                grasp=grasp,
                grasp6d=grasp6d,
                pregrasp6d=pre6d,
                retreat6d=retreat6d,
                pregrasp_joints=pre.joints,
                grasp_joints=grasp_solution.joints,
                retreat_joints=retreat.joints,
                approach_joint_waypoints=approach_waypoints,
                retreat_joint_waypoints=retreat_waypoints,
                flip=flip,
                raw_score=raw_score,
                normalized_score=normalized_score,
                cost=total_cost,
            )
        )

    selected = best_validated_candidate(executable, path_validator, cancel_check=cancel_check, metrics=selection_metrics)
    if selected is not None:
        print(
            f"[G] Selected rank cost={selected.cost:.6f} raw_score={selected.raw_score:.6f} "
            f"depth={float(selected.grasp.depth):.6f}m flip={int(selected.flip)}"
        )
        if robot_backend == "rars01":
            selected_tcp = grasp_driver.grasp_tcp_transform(float(selected.grasp.width))
            print(
                "[G] jaw center in End_link [m]: "
                f"{np.round(selected_tcp[:3, 3], 5).tolist()}"
            )
        return selected

    print(
        f"[G] No executable grasp: depth={skipped_depth} low_z={skipped_low} jaw_z={skipped_jaw} "
        f"ik_fail={skipped_ik} max_err={worst_err:.4f}"
    )
    return None

