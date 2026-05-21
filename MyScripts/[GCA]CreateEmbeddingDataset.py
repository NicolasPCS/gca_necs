import os
import argparse
import numpy as np
from tqdm import tqdm
from MinkowskiEngine.utils import sparse_quantize # type: ignore

# np version of the quantize function in utils.util
def quantize(points, voxel_size):
    coords = np.round(points / voxel_size).astype(np.int32)
    q_coords = sparse_quantize(
        coords, return_index=False, quantization_size=1
    )
    
    if hasattr(q_coords, "cpu"):
        q_coords = q_coords.cpu().numpy()
    
    return q_coords

def create_embeddings(input_file, output_file, voxel_size, random_translation=True):
    data = np.load(input_file)

    if "surface" not in data.files:
        raise ValueError(f"File {input_file} not found.")
    
    surface = data['surface'].astype(np.float32)
    
    if random_translation:
        translation = 4 * np.random.rand(1, 4).astype(np.float32) * voxel_size
    else:
        translation = np.zeros((1, 4), dtype=np.float32)
    
    # From original code: shifts before quantize
    surface_shifted = surface + translation[:, :3]

    y_coords = quantize(surface_shifted, voxel_size)

    # For pure GCA, the features are ignored, that's why here I treat them as occupation
    y_feats = np.ones((y_coords.shape[0], 1), dtype=np.float32)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez_compressed(output_file, coord=y_coords.astype(np.int32), feat=y_feats, translation=translation.astype(np.float32))

    return y_coords.shape[0]

def process_split(args, split):
    dir = os.path.join(args.data_root, args.class_name)
    split_dir = os.path.join(dir, split)
    txt_path = os.path.join(dir, f"{split}.txt")
    output_dir = os.path.join(args.embedding_root, args.class_name, split)

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"{txt_path} does not exist.")

    voxel_counts = []

    for model_id in tqdm(ids):
        input_file = os.path.join(split_dir, f"{model_id}.npz")

        if not os.path.exists(input_file):
            print(f"{input_file} does not exist.")
            continue
            
        if split == "train":
            for aug_id in range(args.num_aug):
                output_file = os.path.join(output_dir, f"{model_id}_{aug_id}.npz")

                n_voxels = create_embeddings(input_file, output_file, args.voxel_size, random_translation=True)

                voxel_counts.append(n_voxels)
        
        elif split == "test":
            output_file = os.path.join(output_dir, f"{model_id}.npz")

            n_voxels = create_embeddings(input_file, output_file, args.voxel_size, random_translation=False)

            voxel_counts.append(n_voxels)
        
        else:
            print("Unsupported split")
            return
    
    voxel_counts = np.array(voxel_counts)

    print(f"\nSummary {split}:")
    print(f"Min voxeles : {voxel_counts.min()}")
    print(f"Max voxeles : {voxel_counts.max()}")
    print(f"Mean voxeles: {voxel_counts.mean():.2f}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", help="Path to the shapenet_sdf dataset")
    parser.add_argument("--embedding_root", help="Path to save embeddings")
    parser.add_argument("--class_name", help="ShapeNet object category")
    parser.add_argument("--voxel_size", type=float, default=0.03125, help="Voxel size")
    parser.add_argument("--num_aug", type=int, default=10, help="Number of embeddings files to generate")

    args = parser.parse_args()

    np.random.seed(49)

    process_split(args, "train")
    process_split(args, "test")

    print("\nDone!")
    print("Embeddings saved in:", os.path.join(args.embedding_root, args.class_name))

if __name__ == "__main__":
    main()