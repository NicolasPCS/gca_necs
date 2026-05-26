import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MinkowskiEngine as ME  # type: ignore  # noqa: E402
from MinkowskiEngine.utils import sparse_quantize  # type: ignore  # noqa: E402
from models import MODEL  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GCA samples from partial ShapeNet objects stored as .npz files."
    )
    parser.add_argument(
        "--config",
        default="/home/isipiran/gca_necs/log/05-15-23:59:44/config.yaml",
        help="Path to the experiment config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/isipiran/gca_necs/log/05-15-23:59:44/ckpts/ckpt-step-492000",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/isipiran/gca_necs/data/shapenet_sdf/chair/test",
        help="Directory containing ShapeNet .npz files with a 'surface' array.",
    )
    parser.add_argument(
        "--output-subdir",
        default="generated_pcs_derived_from_shape",
        help="Output folder name created next to the config file.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=None,
        help="Maximum number of .npz shapes to process. Default: all files in input-dir.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of CA transition steps. Default: config max_eval_phase or max_phase.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points to sample from the final generated shape.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for partial seed creation and point sampling.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save sparse voxels and meshes for intermediate CA steps. Disabled by default.",
    )
    return parser.parse_args()


def quantize(coord, voxel_size):
    coord = torch.round(coord / voxel_size).cpu()
    return sparse_quantize(coord, return_index=False, quantization_size=1).int()


def downsample(coord, n):
    if coord.shape[0] <= n:
        return coord
    idx = torch.randperm(coord.shape[0])[:n]
    return coord[idx]


def create_sphere_seed(npz_path, config, in_channels, device):
    data = np.load(npz_path)
    if "surface" not in data:
        raise KeyError(f"File {npz_path} does not contain a 'surface' array")

    shape_coord = torch.tensor(data["surface"]).float()

    max_sphere_centers = config["max_sphere_centers"]
    sphere_radius = config["sphere_radius"]
    surface_cnt = config["surface_cnt"]

    num_sphere_centers = torch.randint(max_sphere_centers, (1,)).item() + 1
    sphere_centers = shape_coord[
        torch.randint(shape_coord.shape[0], (num_sphere_centers,)), :
    ]

    if len(sphere_centers.shape) == 1:
        sphere_centers = sphere_centers.reshape(1, -1)

    survived_idxs = torch.zeros(shape_coord.shape[0]).bool()

    for center in sphere_centers:
        dists = torch.sqrt(torch.sum((shape_coord - center) ** 2, dim=1))
        survived_idxs = survived_idxs | (dists < sphere_radius)

    point_coord = shape_coord[survived_idxs, :]
    point_coord = downsample(point_coord, surface_cnt)
    state_coord = quantize(point_coord, config["voxel_size"])

    batch = torch.zeros((state_coord.shape[0], 1), dtype=torch.int32)
    coords = torch.cat([batch, state_coord], dim=1).int()
    feats = torch.ones((coords.shape[0], in_channels), dtype=torch.float32)

    return ME.SparseTensor(features=feats, coordinates=coords, device=device)


def extract_pointcloud(model, sparse_state, num_points):
    pointcloud = model.get_pointcloud(sparse_state, [num_points], return_mesh=False)
    if isinstance(pointcloud, dict):
        pointcloud = pointcloud[num_points][0]
    elif isinstance(pointcloud, (list, tuple)):
        pointcloud = pointcloud[0]
    if not isinstance(pointcloud, torch.Tensor):
        pointcloud = torch.tensor(pointcloud)

    pointcloud_np = pointcloud.detach().cpu().numpy().astype(np.float32)
    if pointcloud_np.shape != (num_points, 3):
        raise ValueError(f"Expected point cloud shape ({num_points}, 3), got {pointcloud_np.shape}")
    return pointcloud_np


def save_intermediate_step(model, s, sample_nums, object_idx, step, output_path):
    step_dir = os.path.join(output_path, "intermediate_steps", f"object_{object_idx:04d}")
    os.makedirs(step_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(step_dir, f"step_{step:03d}_voxels.npz"),
        coords=s.C.cpu().numpy(),
        feats=s.F.cpu().numpy(),
    )

    try:
        _, mesh_dict = model.get_pointcloud(s, sample_nums, return_mesh=True)
        for mesh_key, meshes in mesh_dict.items():
            for batch_idx, mesh in enumerate(meshes):
                file_path = os.path.join(
                    step_dir,
                    f"step_{step:03d}_{mesh_key}_{batch_idx}.obj",
                )
                mesh.export(file_path)
    except Exception as exc:
        print(f"  [WARNING] could not save mesh at step {step}: {exc}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = MODEL[config["model"]](config, writer=None)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    num_steps = args.num_steps if args.num_steps is not None else config.get("max_eval_phase", config.get("max_phase", 30))
    in_channels = config["backbone"].get("in_channels", 1)
    voxel_overflow_limit = config.get("voxel_overflow", 20000)
    sample_nums = [args.num_points]

    output_path = os.path.join(os.path.dirname(args.config), args.output_subdir)
    os.makedirs(output_path, exist_ok=True)

    npz_files = sorted(
        os.path.join(args.input_dir, filename)
        for filename in os.listdir(args.input_dir)
        if filename.endswith(".npz")
    )
    if args.num_objects is not None:
        npz_files = npz_files[:args.num_objects]
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {args.input_dir}")

    print("Generating from partial ShapeNet seeds...")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  input_dir: {args.input_dir}")
    print(f"  output_path: {output_path}")
    print(f"  num_shapes: {len(npz_files)}")
    print(f"  save_intermediate: {args.save_intermediate}")

    with torch.no_grad():
        for object_idx, npz_path in enumerate(npz_files):
            shape_id = os.path.splitext(os.path.basename(npz_path))[0]
            print(f"\nObject {object_idx:04d}: {shape_id}")

            s = create_sphere_seed(npz_path, config, in_channels, device)
            print(f"    step 00: {s.C.shape[0]} voxels")
            if args.save_intermediate:
                save_intermediate_step(model, s, sample_nums, object_idx, 0, output_path)

            for step_idx in range(num_steps):
                s = model.transition(s)
                n_voxels = s.C.shape[0]
                print(f"    step {step_idx + 1:02d}: {n_voxels} voxels")

                if args.save_intermediate:
                    save_intermediate_step(model, s, sample_nums, object_idx, step_idx + 1, output_path)

                if n_voxels > voxel_overflow_limit:
                    print(f"  [WARNING] voxel overflow: {n_voxels}")
                    break

            pointcloud = extract_pointcloud(model, s, args.num_points)
            pc_path = os.path.join(output_path, f"generated_object_{object_idx:04d}_{shape_id}.npy")
            np.save(pc_path, pointcloud)

    print("\nDone.")
    print(f"Point clouds saved in: {output_path}")


if __name__ == "__main__":
    main()
