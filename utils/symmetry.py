"""
Reflection symmetry utilities for sparse voxel coordinates.

Coordinates may be either unbatched spatial coordinates [x, y, z] or MinkowskiEngine-style batched coordinates [batch, x, y, z]
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

AXIS_TO_INDEX = {
    'x': 0,
    'y': 1, 
    'z': 2,
    0: 0,
    1: 1,
    2: 2,
}

def get_symmetry_config(config: dict) -> dict:
    """
    Return symmetry config with backward-compatible defaults.
    """
    symmetry = dict(config.get('symmetry', {}) or {})
    defaults = {
        'enabled': False,
        'axis': 'x',
        'plane_value': 0,
        'train_loss_weight': 0.0,
        'train_loss_type': 'occupancy',
        'enforce_sampling': False,
        'enforce_sampling_mode': 'final',
        'merge_features': 'average',
        'debug': False,
    }
    for key, value in defaults.items():
        symmetry.setdefault(key, value)
    return symmetry

def detect_spatial_coordinate_columns(coords: torch.Tensor, data_dim: int = 3, coordinate_layout: str = 'auto') -> Tuple[int, ...]:
    """
    Detect spatial columns.
    
    coordinate_layout='batched' follows MinkowskiEngine coordinates [b, x, y, z] and preserves column 0 as the batch index.
    """

    if coords.ndim != 2:
        raise ValueError('Coords must be 2D, got {}'.format(tuple(coords.shape)))
    
    if coordinate_layout == 'auto':
        if coords.shape[1] == data_dim + 1:
            coordinate_layout = 'batched'
        elif coords.shape[1] == data_dim:
            coordinate_layout = 'spatial'
        else:
            raise ValueError('Cannot infer layout from coords shape {} with data_dim={}'.format(tuple(coords.shape), data_dim))
    
    if coordinate_layout == 'batched':
        print(f"coordinate_layout = {coordinate_layout}. return: {tuple(range(1, data_dim + 1))}")
        return tuple(range(1, data_dim + 1))
    if coordinate_layout == 'spatial':
        print(f"coordinate_layout = {coordinate_layout}. return: {tuple(range(data_dim))}")
        return tuple(range(data_dim))
    raise ValueError("Unknown coordinate_layout {}".format(coordinate_layout))

def _axis_column(coords: torch.Tensor, axis='x', data_dim: int = 3, coordinate_layout: str = 'auto') -> int:
    if axis not in AXIS_TO_INDEX:
        raise ValueError("axis must be x/y/z or 0/1/2, got {}".format(axis))
    axis_index = AXIS_TO_INDEX[axis]
    spatial_columns = detect_spatial_coordinate_columns(coords, data_dim, coordinate_layout)
    if axis_index >= len(spatial_columns):
        raise ValueError('axis {} is invalid fpr data_dim={}'.format(axis, data_dim))
    print(f"spatial_columns = {spatial_columns}. axis_index = {axis_index}")
    return spatial_columns[axis_index]

def reflect_coords(coords: torch.Tensor, axis='x', plane_value=0, coordinate_layout: str = 'auto', data_dim: int =3) -> torch.Tensor:
    """
    Reflect coordinates across axis = plane_value.

    For MinkowskiEngine-style coordinates, only spatial columns are reflected; the batch col is preserved.
    """
    reflected = coords.clone()
    axis_col = _axis_column(coords, axis, data_dim, coordinate_layout)
    plane = torch.as_tensor(plane_value, dtype=coords.dtype, device=coords.device) # tensor [0]
    reflected[:, axis_col] = (2 * plane) - reflected[:, axis_col] # for x = 0: 2*0 - x = -x
    return reflected

def sparse_coord_key(coords: torch.Tensor) -> torch.Tensor:
    """
    Canonical integer tensor key representation for sparse coordinates.
    """
    if coords.ndim != 2:
        raise ValueError('coords must be 2D, got {}'.format(tuple(coords.shape)))
    return coords.long()

def _scatter_mean(feats: torch.Tensor, inverse: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Group the rows of feats according to inverse. Returns the average of features corresponding to each group.
    """
    out = torch.zeros(num_groups, feats.shape[1], dtype=feats.dtype, device=feats.device)
    counts = torch.zeros(num_groups, 1, dtype=feats.dtype, device=feats.device)
    out.index_add_(0, inverse, feats)
    counts.index_add_(0, inverse, torch.ones(feats.shape[0], 1, dtype=feats.dtype, device=feats.device))
    return out / counts.clamp_min(1)

def make_symmetric_sparse_coords(coords: torch.Tensor, feats: Optional[torch.Tensor] = None, axis='x', plane_value=0, merge_features: str = 'average', coordinate_layout: str = 'auto', data_dim: int = 3) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Append reflected coordinates, deduplicate, and optionally merge features.
    """
    reflected = reflect_coords(coords, axis, plane_value, coordinate_layout, data_dim)
    all_coords = torch.cat([coords, reflected], dim=0)
    unique_coords, inverse = torch.unique(sparse_coord_key(all_coords), dim=0, sorted=True, return_inverse=True)
    unique_coords = unique_coords.to(dtype=coords.dtype, device=coords.device)

    if feats is None:
        return unique_coords, None
    if feats.shape[0] != coords.shape[0]:
        raise ValueError('feats and coords must have the same first dimension')
    
    all_feats = torch.cat([feats, feats], dim=0)
    if merge_features == 'average':
        unique_feats = _scatter_mean(all_feats, inverse.to(feats.device), unique_coords.shape[0])
    elif merge_features == 'first':
        unique_feats = torch.zeros(unique_coords.shape[0], feats.shape[1], dtype=feats.dtype, device=feats.device)
        seen = torch.zeros(unique_coords.shape[0], dtype=torch.bool, device=feats.device)
        for idx in range(all_feats.shape[0]):
            group = inverse[idx].item()
            if not seen[group]:
                unique_feats[group] = all_feats[idx]
                seen[group] = True
    else:
        raise ValueError('merge_features {} is not supported'.format(merge_features))
    return unique_coords, unique_feats

def pair_indices(coords: torch.Tensor, axis='x', plane=0, coordinate_layout: str = 'auto', data_dim: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Take each sparse coordinate and reflect it with respect to plane=0, and find the original points that has its symmetric counterpart also present in coords. 
    """
    keys = sparse_coord_key(coords)
    reflected = sparse_coord_key(reflect_coords(coords, axis, plane, coordinate_layout, data_dim))
    _, inverse = torch.unique(torch.cat([keys, reflected], dim=0), dim=0, sorted=True, return_inverse=True)
    num_coords = coords.shape[0]
    coord_ids = inverse[:num_coords]
    reflected_ids = inverse[num_coords:]
    order = torch.argsort(coord_ids)
    sorted_ids = coord_ids[order]
    pos = torch.searchsorted(sorted_ids, reflected_ids)
    clamped_pos = pos.clamp_max(max(sorted_ids.shape[0]-1, 0))
    valid = (pos < sorted_ids.shape[0]) & (sorted_ids[clamped_pos] == reflected_ids)
    left = torch.arange(num_coords, device=coords.device)[valid]
    right = order[pos[valid]]
    keep = left < right
    return left[keep], right[keep]

def compute_symmetry_occupancy_loss(coords: torch.Tensor, occupancy_logits: torch.Tensor, axis='x', plane_value=0, coordinate_layout: str = 'auto', data_dim: int = 3) -> torch.Tensor:
    """
    MSE consistency between logits of existing reflected coordinate pairs.
    """
    if occupancy_logits.ndim > 1:
        occupancy_logits = occupancy_logits.squeeze(-1)
    left, right = pair_indices(coords, axis, plane_value, coordinate_layout, data_dim)
    if left.numel() == 0:
        return occupancy_logits.sum() * 0.0
    return F.mse_loss(occupancy_logits[left], occupancy_logits[right])

def compute_symmetry_feature_loss(coords: torch.Tensor, feats: torch.Tensor, axis='x', plane_value=0, coordinate_layout: str = 'auto', data_dim: int = 3) -> torch.Tensor:
    """
    MSE consistency between features of existing reflected coordinate pairs.
    """
    left, right = pair_indices(coords, axis, plane_value, coordinate_layout, data_dim)
    if left.numel() == 0:
        return feats.sum() * 0.0
    return F.mse_loss(feats[left], feats[right])

def symmetry_error(coords: torch.Tensor, axis='x', plane_value=0, coordinate_layout: str = 'auto', data_dim: int = 3) -> float:
    """
    Unmatched reflected voxel ratio. A perfectly symmetric set gives 0.
    """
    if coords.shape[0] == 0:
        return 0.0
    keys = sparse_coord_key(coords)
    reflected = sparse_coord_key(reflect_coords(coords, axis, plane_value, coordinate_layout, data_dim))
    _, inverse = torch.unique(torch.cat([keys, reflected], dim=0), dim=0, sorted=True, return_inverse=True)
    num_coords = coords.shape[0]
    coord_ids = inverse[:num_coords]
    reflected_ids = inverse[num_coords:]
    order = torch.argsort(coord_ids)
    sorted_ids = coord_ids[order]
    pos = torch.searchsorted(sorted_ids, reflected_ids)
    clampled_pos = pos.clamp_max(max(sorted_ids.shape[0]-1, 0))
    valid = (pos < sorted_ids.shape[0]) & (sorted_ids[clampled_pos] == reflected_ids)
    return 1.0 - valid.float().mean().item()