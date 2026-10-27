"""Pure, testable costs for selecting an executable grasp candidate."""

from __future__ import annotations

import numpy as np


def normalized_grasp_costs(scores: np.ndarray, config=None) -> np.ndarray:
    """Fixed configured score range, or legacy min-max; never a success probability."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    config=config or {}
    mode=config.get('score_normalization','minmax')
    if mode=='fixed':
        low=float(config.get('score_min',0.));high=float(config.get('score_max',1.))
        if not np.isfinite([low,high]).all() or high<=low:raise ValueError('invalid fixed score range')
        return 1.-np.clip((values-low)/(high-low),0.,1.)
    if mode!='minmax':raise ValueError('invalid score normalization')
    span = float(values.max() - values.min())
    if span <= np.finfo(np.float64).eps:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return 1.0 - (values - values.min()) / span


def joint_limit_cost(
    configurations: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Maximum squared normalized distance to the centre of joint limits."""
    q = np.asarray(configurations, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(upper, dtype=np.float64).reshape(1, -1)
    midpoint = 0.5 * (lo + hi)
    normalized = 2.0 * np.abs(q - midpoint) / (hi - lo)
    return float(np.clip(np.max(normalized * normalized), 0.0, 1.0))


def motion_cost(
    configurations: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normalization: float,
    *, smooth: bool = True,
) -> float:
    """Normalized accumulated joint motion for the executed configuration chain."""
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("motion normalization must be finite and positive")
    q = np.asarray(configurations, dtype=np.float64)
    if len(q) < 2:
        return 0.0
    ranges = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    normalized_steps = np.diff(q, axis=0) / ranges.reshape(1, -1)
    distance = float(np.linalg.norm(normalized_steps, axis=1).sum())
    if normalization <= 0.0:
        raise ValueError("motion normalization must be positive")
    return float(distance / (distance + normalization) if smooth else np.clip(distance / normalization, 0.0, 1.0))


def best_validated_candidate(candidates, path_validator=None, *, cancel_check=None, metrics=None):
    """Stable cost order preserves exhaustive filter-then-min selection.

    Costs must be finite and validation must not change candidate costs. A
    timeout/cancellation is propagated, never treated as geometric rejection.
    """
    candidates = list(candidates)
    if any(not np.isfinite(item.cost) for item in candidates):
        raise ValueError("nonfinite grasp candidate cost")
    ranked = sorted(enumerate(candidates), key=lambda pair: pair[1].cost)
    if metrics is not None:
        metrics.update(candidate_count=len(candidates), checked_count=0,
                       selected_original_index=None, selected_cost=None)
    for original_index, candidate in ranked:
        if cancel_check is not None:
            cancel_check()
        if path_validator is not None:
            if metrics is not None:
                metrics['checked_count'] += 1
            if not path_validator(candidate):
                continue
        if metrics is not None:
            metrics.update(selected_original_index=original_index, selected_cost=float(candidate.cost))
        return candidate
    return None


def candidate_cost_terms(chain, lower, upper, grasp_cost, width, max_width, approach, config, *,
                         shape=None, closing_axis=None, ik_errors=None, duration_s=None, center=None):
    """Full-route ranking only; never replaces IK or collision admission."""
    weights = {k: float(config.get('weight_'+k, default)) for k, default in
               [('grasp', .35), ('motion', .30), ('joint', .25), ('width', .10), ('direction', 0.), ('shape',0.), ('ik',0.), ('time',0.), ('turn',0.), ('center',0.)]}
    reserve = float(config.get('desired_width_reserve_m', .010))
    if (not all(np.isfinite(v) and v >= 0 for v in weights.values()) or
            sum(weights.values()) <= 0 or not np.isfinite(reserve) or reserve <= 0 or
            not np.isfinite(max_width) or max_width <= 0 or not np.isfinite(width) or width < 0):
        raise ValueError('invalid candidate selection weights or width reserve')
    terms = dict(grasp=float(grasp_cost),
                 motion=motion_cost(chain, lower, upper, float(config.get('motion_normalization', 2.))),
                 joint=joint_limit_cost(chain, lower, upper),
                 width=float(max(0., (reserve-(max_width-width))/reserve)**2), direction=0.)
    if weights['direction'] > 0:
        preferred = np.asarray(config.get('preferred_approach_base', [0., 0., -1.]), dtype=float)
        actual = np.asarray(approach, dtype=float)
        if (preferred.shape != (3,) or actual.shape != (3,) or
                not np.isfinite(preferred).all() or not np.isfinite(actual).all() or
                np.linalg.norm(preferred) == 0 or np.linalg.norm(actual) == 0):
            raise ValueError('invalid candidate approach direction')
        terms['direction'] = float((1.-np.clip(np.dot(preferred,actual)/
            (np.linalg.norm(preferred)*np.linalg.norm(actual)), -1., 1.))/2.)
    terms.update(shape=0.,ik=0.,time=0.,turn=0.,center=0.)
    if weights['center']>0 and center and center.get('active'):
        value=float(center['cost'])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('invalid center cost')
        terms['center']=value
    if weights['shape']>0 and shape and shape.get('active'):
        axis=np.asarray(shape['axis_camera'],dtype=float);closing=np.asarray(closing_axis,dtype=float)
        terms['shape']=float(shape['confidence']*(axis@closing/(np.linalg.norm(axis)*np.linalg.norm(closing)))**2)
    if weights['ik']>0:
        errors=np.asarray(ik_errors,dtype=float)
        limits=np.array([config['ik_position_scale_m'],np.deg2rad(config['ik_orientation_scale_deg'])])
        if errors.ndim!=2 or errors.shape[1]!=2 or not len(errors) or not np.isfinite(errors).all() or np.any(errors<0) or not np.isfinite(limits).all() or np.any(limits<=0):raise ValueError('invalid IK ranking errors')
        terms['ik']=float(np.mean(np.max(errors/limits,axis=0)**2))
    if weights['time']>0:
        scale=float(config.get('time_scale_s',30.))
        if duration_s is None or not np.isfinite(duration_s) or duration_s<0 or not np.isfinite(scale) or scale<=0:raise ValueError('invalid ranking duration')
        terms['time']=float(duration_s/(duration_s+scale))
    if weights['turn']>0:
        scale=float(config.get('turn_scale_rad',np.pi))
        if not np.isfinite(scale) or scale<=0:raise ValueError('invalid turn scale')
        excursion=float(np.max(np.ptp(np.asarray(chain),axis=0)))
        terms['turn']=float(excursion**2/(excursion**2+scale**2))
    terms['total'] = sum(weights[k]*terms[k] for k in weights)
    return terms
