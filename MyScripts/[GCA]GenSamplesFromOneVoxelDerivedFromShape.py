import os
import sys
import yaml
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODEL
import MinkowskiEngine as ME  # type: ignore
from MinkowskiEngine.utils import sparse_quantize  # type: ignore


config_path = "/home/isipiran/gca_necs/log/05-15-23:59:44/config.yaml"
checkpoint_path = "/home/isipiran/gca_necs/log/05-15-23:59:44/ckpts/ckpt-step-492000"

# CAMBIO: path del objeto .npz
npz_path = "/home/isipiran/gca_necs/data/shapenet_sdf/chair/test/cbe006da89cca7ffd6bab114dd47e3f.npz"

config = yaml.load(open(config_path), Loader=yaml.FullLoader)
model = MODEL[config['model']](config, writer=None)

device = config['device']
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

num_trials = 1
num_steps = config.get("max_eval_phase", config.get("max_phase", 30))
test_sample_nums = config.get('test_sample_nums', [2048])
in_channels = config["backbone"].get("in_channels", 1)


def quantize(coord, voxel_size):
    coord = torch.round(coord / voxel_size).cpu()
    return sparse_quantize(coord, return_index=False, quantization_size=1).int()


def downsample(coord, n):
    if coord.shape[0] <= n:
        return coord
    idx = torch.randperm(coord.shape[0])[:n]
    return coord[idx]


def create_sphere_seed(npz_path):
    data = np.load(npz_path)
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

    s_inicial = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device=device
    )

    return s_inicial

def save_intermediate_step(s, trial, step, output_path):
    step_dir = os.path.join(output_path, f"trial_{trial}")
    os.makedirs(step_dir, exist_ok=True)

    # guarda voxeles sparse
    np.savez_compressed(
        os.path.join(step_dir, f"step_{step:03d}_voxels.npz"),
        coords=s.C.cpu().numpy(),
        feats=s.F.cpu().numpy()
    )

    # guarda mesh
    try:
        _, mesh_dict = model.get_pointcloud(
            s,
            test_sample_nums,
            return_mesh=True
        )

        for k, meshes in mesh_dict.items():
            for batch_idx, mesh in enumerate(meshes):
                file_path = os.path.join(
                    step_dir,
                    f"step_{step:03d}_{k}_{batch_idx}.obj"
                )
                mesh.export(file_path)

    except Exception as e:
        print(f"  [WARNING] no se pudo guardar mesh en step {step}: {e}")

output_path = os.path.join(
    os.path.dirname(config_path),
    "generated_objs_from_sphere_seed"
)
os.makedirs(output_path, exist_ok=True)

print("Generating...")

with torch.no_grad():

    for trial in range(num_trials):
        print(f"\nTrial {trial}")

        s = create_sphere_seed(npz_path)

        # guardar estado inicial
        print(f"    step 00: {s.C.shape[0]} voxels")
        save_intermediate_step(s, trial, 0, output_path)

        for t in range(num_steps):
            s = model.transition(s)

            n_voxels = s.C.shape[0]
            print(f"    step {t + 1:02d}: {n_voxels} voxels")

            # guardar cada paso intermedio
            save_intermediate_step(s, trial, t + 1, output_path)

            if n_voxels > config.get("voxel_overflow", 20000):
                print(f"  [WARNING] voxel overflow: {n_voxels}")
                break

        print("Done!")