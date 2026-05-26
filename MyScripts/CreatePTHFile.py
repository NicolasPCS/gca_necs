import os
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--output_path_samples', type=str, default='')
parser.add_argument('--category', type=str, default='airplane')

args = parser.parse_args()

input_path = args.input_path
output_path_samples = args.output_path_samples
category = args.category

if category == 'airplane' or category == 'plane':
    n_files = 404
elif category == 'car':
    n_files = 346
elif category == 'chair':
    n_files = 637
else:
    raise ValueError('Unknown category {}'.format(category))

num_points = 2048
all_pcs = []
used_files = []


file_list = sorted([f for f in os.listdir(input_path) if f.endswith('.npy')])

for filename in file_list:
    if len(all_pcs) >= n_files:
        break

    file_path = os.path.join(input_path, filename)
    pc = np.load(file_path)

    # LION compute_score expects generated samples as Tensor [B, N, 3] or [B, N, 6].
    # Here we create exactly [B, 2048, 3].
    if pc.ndim != 2 or pc.shape[1] < 3:
        print('Skipping non point-cloud file:', file_path, pc.shape)
        continue
    if pc.shape[0] != num_points:
        print('Skipping file with wrong number of points:', file_path, pc.shape)
        continue

    pc = pc[:, :3].astype(np.float32)
    all_pcs.append(pc)
    used_files.append(filename)

if len(all_pcs) == 0:
    raise RuntimeError('No valid point clouds found in {}'.format(input_path))

if len(all_pcs) < n_files:
    print('WARNING: expected {} clouds, found {}'.format(n_files, len(all_pcs)))

all_pcs = np.stack(all_pcs, axis=0)  # [B, 2048, 3]
samples = torch.from_numpy(all_pcs).float()
torch.save(samples, output_path_samples)

print('Saved:', output_path_samples)
print('samples.shape:', tuple(samples.shape))
print('samples.dtype:', samples.dtype)
print('num_used_files:', len(used_files))
