"""
Code modified from: https://fwilliams.info/point-cloud-utils/sections/cleaning_shapenet/
"""

import os
import numpy as np
import point_cloud_utils as pcu
from scipy import stats
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--class_id", help="ID of the shapenet class")
parser.add_argument("--class_name", help="Name of the the shapenet class")

args = parser.parse_args()

print(args)

# Path to the bech category
source_dir = f"/home/isipiran/gca_necs/data/shapenet_sdf/{args.class_id}"
class_name = args.class_name
output_dir = f"/home/isipiran/gca_necs/data/shapenet_sdf/{class_name}"

# Resolution used to convert shapes to watertight manifolds
# Higher value means better quality and slower
manifold_resolution = 20_000

# Number of points in the volume to sample around the shape
num_vol_points = 500_000

# Number of points on the surface to sample
num_surf_pts = 30_000

def make_partial_clouds(surface, sphere_radius=0.5, partial_points=1024):
    n = surface.shape[0]

    center = surface[np.random.randint(n)]
    dist = np.linalg.norm(surface - center[None, :], axis=1)
    partial = surface[dist < sphere_radius]

    if partial.shape[0] == 0:
        partial = surface
    
    replace = partial.shape[0] < partial_points
    idx = np.random.choice(partial.shape[0], partial_points, replace=replace)

    return partial[idx].astype(np.float32)

def process_split(split_name):
    split_file = os.path.join(source_dir, f"{split_name}.txt")
    out_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(out_split_dir, exist_ok=True)

    if not os.path.exists(split_file):
        print(f"File not found: {split_file}, omiting.")
        return
    
    # Read model's IDs from txt file
    with open(split_file, 'r') as f:
        model_ids = [line.strip() for line in f if line.strip()]
    
    axis0_sizes, axis1_sizes, axis2_sizes = [], [], []
    missing_models = []
    failed_models = []

    print(f"\nProcessing: {split_name.upper()} ({len(model_ids)} listed models)")

    for model_id in model_ids:
        obj_path = os.path.join(source_dir, model_id, "models", "model_normalized.obj")

        if not os.path.exists(obj_path):
            missing_models.append(model_id)
            continue

        try:
            # Load the original mesh
            v, f = pcu.load_mesh_vf(obj_path)

            print("Processed:", model_id)

            # Compute and store dimensions for statistical file
            bbox_min, bbox_max = np.min(v, axis=0), np.max(v, axis=0)
            sizes = bbox_max - bbox_min
            
            max_abs = np.max(np.abs(v))

            #print("model:", model_id)
            #print("bbox_min:", bbox_min)
            #print("bbox_max:", bbox_max)
            #print("sizes:", sizes)
            #print("max_abs:", max_abs)

            if max_abs > 1.0:
                print(f"[WARNING] {model_id} se sale de [-1,1], max_abs={max_abs}")

            axis0_sizes.append(sizes[0])
            axis1_sizes.append(sizes[1])
            axis2_sizes.append(sizes[2])

            # Convert mesh to watertight manifold
            vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution)
            nm = pcu.estimate_mesh_vertex_normals(vm, fm) # Computer vertex normals for watertight mesh

            # Generate random points in the volume around the shape
            # NOTE: ShapeNet shapes are normalized within [-1, 1]^3
            p_vol = (np.random.rand(num_vol_points, 3) - 0.5) * 2.0

            # Compute the SDF of the random points
            sdf, _, _ = pcu.signed_distance_to_mesh(p_vol, vm, fm)

            # Sample points on the surface as face ids and barycentric coordinates
            fid_surf, bc_surf = pcu.sample_mesh_random(vm, fm, num_surf_pts)

            # Compute 3D coordinates and normals of surface samples
            p_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, vm)
            n_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, nm)

            # Adapted for GCA
            # Join (x,y,z) with the SDF value
            # sdf[:, None] converts the 1D array to 2D, so they can be concatenated
            vol_data = np.concatenate([p_vol, sdf[:, None]], axis=-1)

            # Separate positives (outside) and negatives (inside)
            pos_mask = sdf > 0
            sdf_pos = vol_data[pos_mask]
            sdf_neg = vol_data[~pos_mask]

            # Save the resulting npz
            npz_path = os.path.join(out_split_dir, f"{model_id}.npz")

            if split_name == "test":
                partial = make_partial_clouds(p_surf)
                np.savez(npz_path, sdf_pos=sdf_pos, sdf_neg=sdf_neg, surface=p_surf, partial=partial)
            else:
                np.savez(npz_path, sdf_pos=sdf_pos, sdf_neg=sdf_neg, surface=p_surf)
        
        except Exception as e:
            failed_models.append({"id": model_id, "error": str(e)})
    
    if axis0_sizes:
        log_path = os.path.join(output_dir, f"log_{split_name}.txt")
        with open(log_path, 'w') as log_file:
            log_file.write(f"total of {len(axis0_sizes)} filesstats of axis0: {stats.describe(axis0_sizes)}\n")
            log_file.write(f"stats of axis1: {stats.describe(axis1_sizes)}\n")
            log_file.write(f"stats of axis2: {stats.describe(axis2_sizes)}\n")
        
    # Final report
    print(f"\n--- REPORTE FINAL: {split_name.upper()} ---")
    print(f"Total listed in txt   : {len(model_ids)}")
    print(f"Succesfully processed : {len(axis0_sizes)}")
    print(f"Files not found       : {len(missing_models)}")
    print(f"Processing errors     : {len(failed_models)}")

    if failed_models:
        print("\n[!] DETALLE DEL PRIMER FALLO:")
        print(f"ID: {failed_models[0]['id']} | Error: {failed_models[0]['error']}")

#process_split("train")
process_split("test")