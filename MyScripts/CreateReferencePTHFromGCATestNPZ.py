#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a LION-compatible reference .pth from GCA ShapeNet SDF test .npz files."
    )
    parser.add_argument(
        "--input-dir",
        default="/home/isipiran/gca_necs/data/shapenet_sdf/airplane/test",
        help="Directory containing GCA test .npz files with data['surface'].",
    )
    parser.add_argument(
        "--output-path",
        default="/home/isipiran/tmp_gca/reference_gca_airplane_test.pth",
        help="Output .pth path.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=404,
        help="Number of objects to include in the reference PTH.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of surface points per object.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used only when a surface has more than num-points.",
    )
    return parser.parse_args()


def sample_or_validate_surface(surface, num_points, rng, path):
    if surface.ndim != 2 or surface.shape[1] < 3:
        raise ValueError(f"Expected data['surface'] with shape [N, >=3] in {path}, got {surface.shape}")

    surface = surface[:, :3].astype(np.float32)

    if surface.shape[0] == num_points:
        return surface
    if surface.shape[0] > num_points:
        idx = rng.choice(surface.shape[0], size=num_points, replace=False)
        return surface[idx]

    raise ValueError(
        f"Surface in {path} has only {surface.shape[0]} points; "
        f"cannot create {num_points} points without replacement."
    )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    rng = np.random.default_rng(args.seed)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {input_dir}")
    if len(npz_files) < args.num_objects:
        print(f"WARNING: requested {args.num_objects} objects, but only found {len(npz_files)}")

    pcs = []
    used_files = []

    for npz_path in npz_files[:args.num_objects]:
        data = np.load(npz_path)
        if "surface" not in data:
            raise KeyError(f"File {npz_path} does not contain data['surface']")

        pc = sample_or_validate_surface(data["surface"], args.num_points, rng, npz_path)
        pcs.append(pc)
        used_files.append(npz_path.name)

    if len(pcs) == 0:
        raise RuntimeError("No valid point clouds were loaded.")

    reference = torch.from_numpy(np.stack(pcs, axis=0)).float()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference, output_path)

    print("Saved:", output_path)
    print("reference.shape:", tuple(reference.shape))
    print("reference.dtype:", reference.dtype)
    print("num_used_files:", len(used_files))
    print("input_dir:", input_dir)


if __name__ == "__main__":
    main()
