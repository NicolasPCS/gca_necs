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

config_path = "/home/isipiran/gca_necs/log/05-15-23:59:44/config.yaml"
checkpoint_path = "/home/isipiran/gca_necs/log/05-15-23:59:44/ckpts/ckpt-step-492000"

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

num_trials = 5
num_steps = config.get("max_eval_phase", config.get("max_phase", 30))
test_sample_nums = config.get("test_sample_nums", [2048])
voxel_size = config["voxel_size"]
in_channels = config["backbone"].get("in_channels", 1)

output_path = os.path.join(
    os.path.dirname(config_path),
    "generated_objs_from_partial"
)

os.makedirs(output_path, exist_ok=True)


# =========================
# Create initial SparseTensor from partial
# =========================

def create_seed_from_partial():
    data = np.load(data_path)

    if "partial" not in data.files:
        raise KeyError(f"El archivo no tiene partial: {data_path}")

    partial = torch.tensor(data["partial"]).float()

    embedding = np.load(embedding_path)
    translation = torch.tensor(embedding["translation"][:, :3]).float()

    # Igual que en TransitionShapenetDataset:
    # point_coord = point_coord + embedding['translation'][:, :3]
    partial = partial + translation

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

    return s


# =========================
# Generate
# =========================

print("Generating from partial...")
print("data_path:", data_path)
print("embedding_path:", embedding_path)
print("num_steps:", num_steps)
print("test_sample_nums:", test_sample_nums)

with torch.no_grad():

    for trial in range(num_trials):
        print(f"\nTrial {trial}")

        s = create_seed_from_partial()

        print(f"Initial voxels: {s.C.shape[0]}")

        for t in range(num_steps):
            s = model.transition(s)

            n_voxels = s.C.shape[0]
            print(f"    step {t + 1:02d}: {n_voxels} voxels")

            if n_voxels > config.get("voxel_overflow", 20000):
                print(f"  [WARNING] voxel overflow: {n_voxels}")
                break

        # Guardar voxeles finales
        coords_np = s.C[:, 1:].detach().cpu().numpy().astype(np.int32)

        np.savez_compressed(
            os.path.join(output_path, f"{file_id}_trial_{trial}_voxels.npz"),
            coord=coords_np,
            voxel_size=np.array([voxel_size], dtype=np.float32)
        )

        # Convertir a mesh / point cloud
        s_pc_dict, mesh_dict = model.get_pointcloud(
            s,
            test_sample_nums,
            return_mesh=True
        )

        for k, meshes in mesh_dict.items():
            for batch_idx, mesh in enumerate(meshes):
                file_path = os.path.join(
                    output_path,
                    f"{file_id}_trial_{trial}_{k}_{batch_idx}.obj"
                )
                mesh.export(file_path)
                print("Saved mesh:", file_path)

print("\nDone.")