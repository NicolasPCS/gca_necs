import os
import sys
import yaml
import torch
import numpy as np
import MinkowskiEngine as ME  # type: ignore

repo_root = "/home/isipiran/gca_necs"

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import MODEL
from utils.util import quantize


# =========================
# Paths
# =========================

config_path = "/home/isipiran/gca_necs/log/gca_chair/config.yaml"
checkpoint_path = "/home/isipiran/gca_necs/log/gca_chair/ckpts/ckpt-step-300000"

# Cambia este ID por un archivo real de tu test set
class_name = "chair"
file_id = "cbc47018135fc1b1462977c6d3c24550"

data_path = f"/home/isipiran/gca_necs/data/shapenet_sdf/{class_name}/test/{file_id}.npz"

# Debe ser el embedding correspondiente al mismo file_id
embedding_path = f"/home/isipiran/gca_necs/data/embeddings/chair-vox_64-sdf-step_500k/{class_name}/test/{file_id}.npz"


# =========================
# Load config and model
# =========================

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config["device"]

model = MODEL[config["model"]](config, writer=None)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()


# =========================
# Parameters
# =========================

num_trials = 2
num_steps = config.get("max_eval_phase", config.get("max_phase", 30))
test_sample_nums = config.get("test_sample_nums", [2048])
voxel_size = config["voxel_size"]
in_channels = config["backbone"].get("in_channels", 1)
partial_screenshot = True
save_intermediate_steps = False
intermediate_step_interval = 1

output_path = os.path.join(
    os.path.dirname(config_path),
    "generated_npy_from_partial"
)

os.makedirs(output_path, exist_ok=True)


# =========================
# Create initial SparseTensor from partial
# =========================

def create_seed_from_partial():
    data = np.load(data_path)

    if "partial" not in data.files:
        raise KeyError(f"El archivo no tiene partial: {data_path}")
    if "surface" not in data.files:
        raise KeyError(f"El archivo no tiene surface/original: {data_path}")

    partial = torch.tensor(data["partial"]).float()
    original = torch.tensor(data["surface"]).float()

    embedding = np.load(embedding_path)
    translation = torch.tensor(embedding["translation"][:, :3]).float()

    # Igual que en TransitionShapenetDataset:
    # point_coord = point_coord + embedding['translation'][:, :3]
    partial = partial + translation
    original = original + translation

    # Igual que en el repo:
    # state_coord = quantize(point_coord, voxel_size)
    state_coord = quantize(partial, voxel_size).int()

    # Agregar batch index para MinkowskiEngine: [batch, x, y, z]
    coords = ME.utils.batched_coordinates([state_coord]).int()

    feats = torch.ones((coords.shape[0], in_channels), dtype=torch.float32)

    s = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device=device
    )

    return s, partial, original


def save_sparse_coords_npy(sparse_tensor, file_path):
    coords_np = sparse_tensor.C[:, 1:].detach().cpu().numpy().astype(np.int32)
    np.save(file_path, coords_np)
    print("Saved voxel coords:", file_path)


def save_pointclouds_npy(pointcloud_dict, trial, prefix):
    for sample_num, pointclouds in pointcloud_dict.items():
        for batch_idx, pointcloud in enumerate(pointclouds):
            pc_np = pointcloud.detach().cpu().numpy().astype(np.float32)
            file_path = os.path.join(
                output_path,
                f"{file_id}_trial_{trial}_{prefix}_points_{sample_num}_{batch_idx}.npy"
            )
            np.save(file_path, pc_np)
            print("Saved point cloud:", file_path)


# =========================
# Generate
# =========================

print("Generating from partial...")
print("data_path:", data_path)
print("embedding_path:", embedding_path)
print("num_steps:", num_steps)
print("test_sample_nums:", test_sample_nums)
print("save_intermediate_steps:", save_intermediate_steps)
print("intermediate_step_interval:", intermediate_step_interval)

with torch.no_grad():

    for trial in range(num_trials):
        print(f"\nTrial {trial}")

        s, seed_partial, original_surface = create_seed_from_partial()

        print(f"Initial voxels: {s.C.shape[0]}")

        if partial_screenshot:
            seed_path = os.path.join(
                output_path,
                f"{file_id}_trial_{trial}_seed_partial_points.npy"
            )
            np.save(seed_path, seed_partial.detach().cpu().numpy().astype(np.float32))
            print("Saved initial seed point cloud:", seed_path)

        original_path = os.path.join(
            output_path,
            f"{file_id}_trial_{trial}_original_surface_points.npy"
        )
        np.save(original_path, original_surface.detach().cpu().numpy().astype(np.float32))
        print("Saved original object point cloud:", original_path)

        if save_intermediate_steps:
            intermediate_dir = os.path.join(
                output_path,
                f"{file_id}_trial_{trial}_intermediate_steps"
            )
            os.makedirs(intermediate_dir, exist_ok=True)
            save_sparse_coords_npy(
                s,
                os.path.join(intermediate_dir, "step_000_seed_voxels.npy")
            )

        for t in range(num_steps):
            s = model.transition(s)

            n_voxels = s.C.shape[0]
            print(f"    step {t + 1:02d}: {n_voxels} voxels")

            if (
                save_intermediate_steps
                and intermediate_step_interval > 0
                and (t + 1) % intermediate_step_interval == 0
            ):
                save_sparse_coords_npy(
                    s,
                    os.path.join(intermediate_dir, f"step_{t + 1:03d}_voxels.npy")
                )

            if n_voxels > config.get("voxel_overflow", 20000):
                print(f"  [WARNING] voxel overflow: {n_voxels}")
                break

        save_sparse_coords_npy(
            s,
            os.path.join(output_path, f"{file_id}_trial_{trial}_final_voxels.npy")
        )

        if hasattr(model, 'apply_final_sampling_symmetry'):
            s = model.apply_final_sampling_symmetry(s)

        # Convertir a nube de puntos final sin exportar mallas OBJ.
        s_pc_dict = model.get_pointcloud(
            s,
            test_sample_nums,
            return_mesh=False
        )

        save_pointclouds_npy(s_pc_dict, trial, "final")

print("\nDone.")
