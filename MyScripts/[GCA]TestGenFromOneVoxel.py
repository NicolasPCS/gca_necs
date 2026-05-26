#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

# Add project root to sys.path to avoid ModuleNotFoundError when running from MyScripts.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import symmetry-measurement utilities from the external protocol repo.
SYMMETRY_PROTOCOL_ROOT = "/home/isipiran/Symmetrization-of-3D-Generative-Models/Symmetry_Measurement_Protocol"
sys.path.append(SYMMETRY_PROTOCOL_ROOT)

import MinkowskiEngine as ME  # type: ignore  # noqa: E402
from ChamferDistance import chamfer_distance  # noqa: E402
from Householder_transform import householder_transformation  # noqa: E402
from models import MODEL  # noqa: E402

def create_single_voxel_seed(in_channels: int, device: torch.device) -> ME.SparseTensor:
    """Create a one-voxel MinkowskiEngine SparseTensor seed at the origin.

    Coordinate shape is [N, 4] for 3D MinkowskiEngine coordinates:
        [batch_index, x, y, z]
    Feature shape is [N, in_channels].
    """
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)
    feats = torch.ones((1, in_channels), dtype=torch.float32, device=device)
    return ME.SparseTensor(features=feats, coordinates=coords, device=device)


def run_ca_generation(
        model,
        in_channels: int,
        device: torch.device,
        num_steps: int,
        voxel_overflow_limit: int,
) -> ME.SparseTensor:
    """Run the GCA transition chain from one voxel and return the final sparse state."""
    s = create_single_voxel_seed(in_channels, device)
    for step_idx in range(num_steps):
        s = model.transition(s)
        n_voxels = s.C.shape[0]
        if n_voxels > voxel_overflow_limit:
            print(
                f"  [WARN] Voxel overflow at step {step_idx}: "
                f"{n_voxels} > {voxel_overflow_limit}. Stopping this trial."
            )
            break
    return s


def extract_2048_pointcloud(model, sparse_state: ME.SparseTensor) -> np.ndarray:
    """Extract exactly one 2048-point cloud from the sparse state.

    model.get_pointcloud(..., return_mesh=False) returns a dictionary:
        {2048: [Tensor(2048, 3)]}
    for batch size 1. The returned NumPy array has shape [2048, 3].
    """
    pointcloud = model.get_pointcloud(sparse_state, [2048], return_mesh=False)
    if isinstance(pointcloud, dict):
        pointcloud = pointcloud[2048][0]
    elif isinstance(pointcloud, (list, tuple)):
        pointcloud = pointcloud[0]
    if not isinstance(pointcloud, torch.Tensor):
        pointcloud = torch.tensor(pointcloud)
    pointcloud_np = pointcloud.detach().cpu().numpy().astype(np.float32)
    if pointcloud_np.shape != (2048, 3):
        raise ValueError(f"Expected point cloud shape (2048, 3), got {pointcloud_np.shape}")
    return pointcloud_np


def compute_householder_chamfer(pointcloud: np.ndarray) -> float:
    """Compute Chamfer distance between a point cloud and its Householder reflection."""
    reflected = householder_transformation(pointcloud)
    return float(chamfer_distance(pointcloud, reflected))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple GCA samples from a one-voxel seed, evaluate each "
            "trial with Householder-reflection Chamfer distance, and save the "
            "best 2048-point cloud for each generated object."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--num-objects", type=int, default=1000)
    parser.add_argument("--trials-per-object", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-trials",
        action="store_true",
        help="Enable trial-based generation and keep the point cloud with lowest symmetry Chamfer. Disabled by default.",
    )
    parser.add_argument("--output-subdir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if args.output_subdir is None:
        args.output_subdir = "generated_pcs" if args.use_trials else "generated_pcs_notrials"
    output_dir = os.path.join(os.path.dirname(args.config), args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and checkpoint...")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  output_dir: {output_dir}")

    model = MODEL[config["model"]](config, writer=None)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    num_steps = args.num_steps if args.num_steps is not None else config.get("max_eval_steps", config.get("max_phase", 30))
    in_channels = config["backbone"].get("in_channels", 1)
    voxel_overflow_limit = config.get("voxel_overflow", 20000)

    if not args.use_trials:
        print(f"Generating {args.num_objects} objects without trials.")

        with torch.no_grad():
            for object_idx in range(args.num_objects):
                print(f"\n=== Object {object_idx:04d} ===")
                sparse_state = run_ca_generation(
                    model=model,
                    in_channels=in_channels,
                    device=device,
                    num_steps=num_steps,
                    voxel_overflow_limit=voxel_overflow_limit,
                )
                pointcloud = extract_2048_pointcloud(model, sparse_state)
                pc_name = f"generated_object_{object_idx:04d}.npy"
                pc_path = os.path.join(output_dir, pc_name)
                np.save(pc_path, pointcloud)

        print("\nDone.")
        print(f"Point clouds saved in: {output_dir}")
        return

    chamfer_scores = np.full((args.num_objects, args.trials_per_object), np.nan, dtype=np.float64)
    best_chamfers = np.full(args.num_objects, np.nan, dtype=np.float64)
    best_trial_indices = np.full(args.num_objects, -1, dtype=np.int64)

    print(
        f"Generating {args.num_objects} objects with "
        f"{args.trials_per_object} trials each."
    )

    with torch.no_grad():
        for object_idx in range(args.num_objects):
            best_pc = None
            best_chamfer = float("inf")
            best_trial = -1

            print(f"\n=== Object {object_idx:04d} ===")
            for trial_idx in range(args.trials_per_object):
                sparse_state = run_ca_generation(
                    model=model,
                    in_channels=in_channels,
                    device=device,
                    num_steps=num_steps,
                    voxel_overflow_limit=voxel_overflow_limit,
                )
                pointcloud = extract_2048_pointcloud(model, sparse_state)
                chamfer = compute_householder_chamfer(pointcloud)
                chamfer_scores[object_idx, trial_idx] = chamfer

                print(f"  trial {trial_idx:02d}: chamfer={chamfer:.8f}")
                if chamfer < best_chamfer:
                    best_chamfer = chamfer
                    best_trial = trial_idx
                    best_pc = pointcloud

            if best_pc is None:
                raise RuntimeError(f"No valid point cloud generated for object {object_idx}")

            best_chamfers[object_idx] = best_chamfer
            best_trial_indices[object_idx] = best_trial

            pc_name = f"generated_object_{object_idx:04d}_best_trial_{best_trial:02d}_cd_{best_chamfer:.8f}.npy"
            pc_path = os.path.join(output_dir, pc_name)
            np.save(pc_path, best_pc)

    print("\nDone.")
    print(f"Best point clouds and Chamfer arrays saved in: {output_dir}")


if __name__ == "__main__":
    main()
