import argparse
import torch


def load_pcs(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        pcs = obj['ref']
        stored_mean = obj.get('mean')
        stored_std = obj.get('std')
    else:
        pcs = obj
        stored_mean = None
        stored_std = None
    return pcs[:, :, :3].float(), stored_mean, stored_std


def stats(name, pcs, stored_mean=None, stored_std=None):
    per_shape_mean = pcs.mean(dim=1, keepdim=True)          # [B, 1, 3]
    per_shape_std = pcs.reshape(pcs.shape[0], -1).std(dim=1).view(-1, 1, 1)  # [B, 1, 1]
    bbox_min = pcs.amin(dim=1)
    bbox_max = pcs.amax(dim=1)
    extent = bbox_max - bbox_min

    print('\n== {} =='.format(name))
    print('shape:', tuple(pcs.shape))
    print('global_mean:', pcs.mean(dim=(0, 1)).tolist())
    print('global_std:', pcs.reshape(-1, 3).std(dim=0).tolist())
    print('per_shape_mean_mean:', per_shape_mean.mean(dim=0).squeeze(0).tolist())
    print('per_shape_mean_abs_mean:', per_shape_mean.abs().mean(dim=0).squeeze(0).tolist())
    print('per_shape_std_mean:', per_shape_std.mean().item())
    print('bbox_min_global:', bbox_min.amin(dim=0).tolist())
    print('bbox_max_global:', bbox_max.amax(dim=0).tolist())
    print('extent_mean:', extent.mean(dim=0).tolist())
    print('max_extent_mean:', extent.max(dim=1).values.mean().item())

    if stored_mean is not None:
        print('stored_mean_shape:', tuple(stored_mean.shape))
        print('stored_mean_mean:', stored_mean.float().mean(dim=0).reshape(-1).tolist())
    if stored_std is not None:
        print('stored_std_shape:', tuple(stored_std.shape))
        print('stored_std_mean:', stored_std.float().mean().item())

    return per_shape_mean, per_shape_std, extent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, help='Path to generated samples .pth')
    parser.add_argument('--ref', required=True, help='Path to reference .pth')
    args = parser.parse_args()

    samples, samples_mean, samples_std = load_pcs(args.samples)
    ref, ref_mean, ref_std = load_pcs(args.ref)

    samples_ps_mean, samples_ps_std, samples_extent = stats('samples', samples, samples_mean, samples_std)
    ref_ps_mean, ref_ps_std, ref_extent = stats('ref', ref, ref_mean, ref_std)

    n = min(samples.shape[0], ref.shape[0])
    print('\n== comparison first {} shapes =='.format(n))
    print('mean_abs_diff_global:', (samples[:n].mean(dim=(0, 1)) - ref[:n].mean(dim=(0, 1))).abs().tolist())
    print('per_shape_mean_abs_diff:', (samples_ps_mean[:n] - ref_ps_mean[:n]).abs().mean(dim=0).squeeze(0).tolist())
    print('per_shape_std_abs_diff:', (samples_ps_std[:n] - ref_ps_std[:n]).abs().mean().item())
    print('extent_abs_diff_mean:', (samples_extent[:n] - ref_extent[:n]).abs().mean(dim=0).tolist())
    print('max_extent_abs_diff_mean:', (samples_extent[:n].max(dim=1).values - ref_extent[:n].max(dim=1).values).abs().mean().item())


if __name__ == '__main__':
    main()
