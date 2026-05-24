"""
Deterministic symmetry rules for sparse GCA states.

The rule is applied to the sparse active state. During trainig, the next state that is pushed back into the CA buffer can be closed under a symmetry group. During sampling, the sampled state can be closed under the same rules at every transition step.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch

AXIS_TO_SPATIAL_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    0: 0,
    1: 1,
    2: 2,
}

@dataclass(frozen=True)
class SymmetryRuleSpec:
    """
    Configuration for one object's deterministic symmetry rule.

    type: reflection, rotation, reflection_rotation, or none
    axis: axis used by reflection or rotation. For refelction, axis=x means plane x=plane_value. For rotation, axis=z means rotate around the z-axis.
    plane_value: Integer voxel coordinate of the reflection plane.
    center: Integer voxel center of rotation. For ShapeNet is [0,0,0].
    order: Rotational symmetry order. 4 menas 0, 90, 180, 270 degrees.
    merge_features: How to merge duplicate features after symmetry closure.
    """
    enabled: bool = False
    type: str = "none"
    axis: str = "x"
    plane_value: int = 0
    center: Tuple[int, int, int] = (0, 0, 0)
    order: int = 2
    merge_features: str = "average"

def get_symmetry_rules_config(config):
    """
    Return deterministic symmetry-rule config with safe defaults.
    """
    rules = dict(config.get("symmetry_rules", {}) or {})
    rules.setdefault("enabled", False)
    rules.setdefault("apply_during_training", False)
    rules.setdefault("apply_during_sampling", False)
    rules.setdefault("debug", False)
    rules.setdefault("metadata_key", "symmetry")
    rules.setdefault("rule", {})
    return rules

def spec_from_dict(rule_dict):
    """
    Convert a config dictionary into a typed SymmetryRuleSpec.
    """
    if not rule_dict:
        return SymmetryRuleSpec(enabled=False)
    center = rule_dict.get("center", [0,0,0])
    if len(center) != 3:
        raise ValueError("rotation center mush have lenght 3, got {}".format(center))
    return SymmetryRuleSpec(
        enabled=bool(rule_dict.get("enabled", True)),
        type = rule_dict.get("type", "none"),
        axis = rule_dict.get("axis", "x"),
        plane_value = int(rule_dict.get("plane_value", 0)),
        center = tuple(int(v) for v in center),
        order = int(rule_dict.get("order", 2)),
        merge_features = rule_dict.get("merge_features", "average")
    )

def batch_specs_from_config(config, batch_size, data=None):
    """
    Build one symmetry spec per batch item.

    If the dataset later provides per-object metadata in data[metadata_key], this function can consume it. Until then, the global config rule is repeated for all batch elements.
    """
    rules_config = get_symmetry_rules_config(config)
    metadata_key = rules_config["metadata_key"]
    if data is not None and metadata_key in data:
        metadata = data[metadata_key]
        if len(metadata) != batch_size:
            raise ValueError("Expected {} metadata items, got {}".format(batch_size, len(metadata)))
        return [spec_from_dict(item) for item in metadata]
    default_spec = spec_from_dict(rules_config.get("rule"))
    return [default_spec for _ in range(batch_size)]

def _spatial_axis_column(axis, data_dim, batched):
    """
    Return the tensor column for spatial axis.

    For coords shape [N, 4] = [bs, x, y, z], x is column 1.
    For coords shape [N, 3] = [x, y, z], x is column 0.
    """
    if axis not in AXIS_TO_SPATIAL_INDEX:
        raise ValueError("Invalid axis {}".format(axis))
    axis_idx = AXIS_TO_SPATIAL_INDEX[axis]
    if axis_idx >= data_dim:
        raise ValueError("Axis {} is invalid for data_dim {}".format(axis, data_dim))
    return axis_idx + 1 if batched else axis_idx

def reflect_batched_coords(coords, axis="x", plane_value=0, data_dim=3):
    """
    Reflect batched sparse coordiantes accross one axis-aligned plane-

    coords: [bs, x, y, z]

    Return reflected coordinates with the same shape and dtype.
    """
    if coords.ndim != 2 or coords.shape[1] != data_dim + 1:
        raise ValueError("Expected batched coords shape [N, {}], got {}".format(data_dim+1, tuple(coords.shape)))
    reflected = coords.clone()
    axis_col = _spatial_axis_column(axis, data_dim, batched=True)
    plane = torch.as_tensor(plane_value, dtype=coords.dtype, device=coords.device)
    reflected[:, axis_col] = (2 * plane) - reflected[:, axis_col]
    return reflected

def rotate_batched_coords_90(coords, axis="z", k=1, center=(0,0,0), data_dim=3):
    """
    Rotate batched 3D voxel coordinates by k * 90 degrees.
    This keeps coordiantes integer exactly. Supports axis x, y, z.
    Shape:
        coords: [N, 4] = [bs, x, y, z]
        return: [N, 4]
    """
    if data_dim != 3:
        raise ValueError("Rotational rules currently require data_dim=3")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Expected coords shape [N, 4], got {}".format(tuple(coords.shape)))
    k = k % 4
    rotated = coords.clone()
    if k == 0:
        return rotated
    
    center_tensor = torch.tensor(center, dtype=coords.dtype, device=coords.device).view(1, 3)
    spatial = coords[:, 1:4] - center_tensor
    x, y, z = spatial[:, 0], spatial[:, 1], spatial[:, 2]
    if axis == "z":
        if k == 1: # NOTE: Rotate 90 degrees
            new_spatial = torch.stack([-y, x, z], dim=1)
        elif k == 2: # NOTE: Rotate 180 degrees
            new_spatial = torch.stack([-x, -y, z], dim=1)
        else: # NOTE: Rotate 270 degrees
            new_spatial = torch.stack([y, -x, z], dim=1)
    elif axis == "y":
        if k == 1:
            new_spatial = torch.stack([z, y, -x], dim=1)
        elif k == 2:
            new_spatial = torch.stack([-x, y, -z], dim=1)
        else:
            new_spatial = torch.stack([-z, y, x], dim=1)
    elif axis == "x":
        if k == 1:
            new_spatial = torch.stack([x, -z, y], dim=1)
        elif k == 2:
            new_spatial = torch.stack([x, -y, -z], dim=1)
        else:
            new_spatial = torch.stack([x, z, -y], dim=1)
    else:
        raise ValueError("Rotation axis must be x, y, or z, got {}".format(axis))

    rotated[:, 1:4] = new_spatial + center_tensor
    return rotated

def transform_coords_by_spec(coords, spec, data_dim=3):
    """
    Return all transformed coordinate sets implied by one rule spec.

    The first returned tensor is always the original coords. Additional tensors are reflections and/or rotations. Concatenating and deduplicating them gives the closure of the active CA state under the selected symmetry rule.
    """
    if not spec.enabled or spec.type in ["none", None]:
        return [coords]
    transforms = [coords]
    if spec.type in ["reflection", "reflection_rotation"]:
        transforms.append(reflect_batched_coords(coords, axis=spec.axis, plane_value=spec.plane_value, data_dim=data_dim))
    if spec.type in ["rotation", "reflection_rotation"]:
        if spec.order not in [1, 2, 4]:
            raise ValueError("Only exact integer rotation orders 1, 2, and 4 are supported; got {}".format(spec.order))
        step = 4 // spec.order
        for k in range(step, 4, step):
            transforms.append(rotate_batched_coords_90(coords, axis=spec.axis, k=k, center=spec.center, data_dim=data_dim))
    return transforms

def _scatter_mean(feats, inverse, num_groups):
    """
    Average duplicate features after coordinate duplication.

    feats = [N_total, C]
    inverse_shape = [N_total], maps each row to a unique coordinate index.
    return shape [num_groups, C]
    """
    out = torch.zeros(num_groups, feats.shape[1], dtype=feats.dtype, device=feats.device)
    counts = torch.zeros(num_groups, 1, dtype=feats.dtype, device=feats.device)
    out.index_add_(0, inverse, feats)
    counts.index_add_(0, inverse, torch.ones(feats.shape[0], 1, dtype=feats.dtype, device=feats.device))
    return out / counts.clamp_min(1)

def deduplicate_coords_and_feats(coords, feats=None, merge_features="average"):
    """
    Deduplicate sparse coords and optionally merge features.
    coords shape: [N, 1+data_dim]
    feats shape: [N, C] or None
    """
    unique_coords, inverse = torch.unique(coords.long(), dim=0, sorted=True, return_inverse=True)
    unique_coords = unique_coords.to(dtype=coords.dtype, device=coords.device)
    if feats is None:
        return unique_coords, None
    if merge_features == "average":
        return unique_coords, _scatter_mean(feats, inverse.to(feats.device), unique_coords.shape[0])
    if merge_features == "first":
        unique_feats = torch.zeros(unique_coords.shape[0], feats.shape[1], dtype=feats.dtype, device=feats.device)
        seen = torch.zeros(unique_coords.shape[0], dtype=torch.bool, device=feats.device)
        for row_idx in range(feats.shape[0]):
            group_idx = inverse[row_idx].item()
            if not seen[group_idx]:
                unique_feats[group_idx] = feats[row_idx]
                seen[group_idx] = True
        return unique_coords, unique_feats
    raise ValueError("Unsupported merge_features {}".format(merge_features))

def apply_symmetry_rule_to_batched_state(coords, feats, batch_specs, data_dim=3):
    """
    Apply per-batch symmetry closure to a sparse active CA state.

    Returns:
        new_coords: Deduplicated coordinates closed under each rule.
        num_feats: Merged features if feats was provided, else None.
    """
    if coords.numel() == 0:
        return coords, feats
    all_coords, all_feats = [], []
    for batch_idx, spec in enumerate(batch_specs):
        mask = coords[:, 0] == batch_idx
        batch_coords = coords[mask]
        if batch_coords.shape[0] == 0:
            continue
        transformed = transform_coords_by_spec(batch_coords, spec, data_dim=data_dim)
        merged_coords = torch.cat(transformed, dim=0)
        all_coords.append(merged_coords)
        if feats is not None:
            batch_feats = feats[mask]
            # Same feature is copied to each symmetry-generated coordinate.
            all_feats.append(batch_feats.repeat(len(transformed), 1))
    
    if not all_coords:
        return coords, feats
    cat_coords = torch.cat(all_coords, dim=0)
    cat_feats = torch.cat(all_feats, dim=0) if feats is not None else None
    merge_features = batch_specs[0].merge_features if batch_specs else "average"
    return deduplicate_coords_and_feats(cat_coords, cat_feats, merge_features=merge_features)

def symmetry_rule_error(coords, spec, data_dim=3):
    """
    Diagnostic: fraction of transformed coordinates missing from coords.
    This is only a metric/debug helper. It is not a training loss.
    """
    if coords.shape[0] == 0 or not spec.enabled:
        return 0.0
    closed_coords, _ = apply_symmetry_rule_to_batched_state(coords, None, [spec], data_dim=data_dim)
    return 1.0 - (coords.shape[0] / float(closed_coords.shape[0]))