import os
import cv2
import numpy as np
import torch
from field_of_junctions import FieldOfJunctions
import matplotlib.pyplot as plt
from tqdm import tqdm
from stereo_BIG_BLOCKS_SAVE_RIGHT import create_opts
from stereo_BIG_BLOCKS_SAVE_RIGHT import get_LRC_mask
from metrics_utils import compute_disparity_metrics_with_window

BOUNDARY_THRESHOLD = 0.1
LRC_THRESHOLD = 1
WINDOW_SIZE = 19

def patches2img(foj, patches):
    """
    Args:
        patches: [C, R, R, H', W']
    Returns:
        img: [H, W, C]
    """
    return foj.local2global(patches.unsqueeze(0))[0].permute(1, 2, 0).detach().cpu().numpy()


def get_all_instereo_paths(base_dir):
    dirs = sorted(os.listdir(base_dir))
    for dir in dirs:
        im0_path = os.path.join(base_dir, f'{dir}/0_L.png')
        im1_path = os.path.join(base_dir, f'{dir}/0_R.png')
        disp0_path = os.path.join(base_dir, f'{dir}/0_disp.npy')
        if not os.path.exists(im0_path) or not os.path.exists(im1_path) or not os.path.exists(disp0_path):
            continue

        yield im0_path, im1_path, disp0_path


def process_one_instereo_pair(im0_path, im1_path, disp0_path, opts):
    left = cv2.imread(im0_path)
    right = cv2.imread(im1_path)
    gt_disp_raw = np.load(disp0_path)

    gt_disp = gt_disp_raw.astype(np.float32)

    # resize to valid size
    H, W, C = left.shape
    valid_x = int((W - opts.R) / opts.stride) * opts.stride + opts.R
    valid_y = int((H - opts.R) / opts.stride) * opts.stride + opts.R
    left = left[:valid_y, :valid_x]
    right = right[:valid_y, :valid_x]
    gt_crop = gt_disp[:valid_y, :valid_x]

    left_norm = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    right_norm = cv2.cvtColor(right, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    foj = FieldOfJunctions(left_norm, opts, right_img=right_norm)

    foj.optimize()

    return foj, gt_crop


def visualize(foj, opts, disp0_path, save_path):
    H, W = foj.H, foj.W
    gt_disp = np.load(disp0_path)
    gt_disp = gt_disp.astype(np.float32)
    gt_crop = gt_disp[:H, :W]

    gt_for_window = gt_crop.copy()
    gt_for_window[(gt_for_window == 0)] = np.nan

    pred_disp_final = foj.get_disparity_map() * (opts.R / 2.0) # final disparity is in normalized coordinates
    pred_disp_right = foj.get_right_disparity_patches() * (opts.R / 2.0)  # [1, R, R, H', W']
    pred_disp_right = foj.local2global(pred_disp_right.unsqueeze(0))[0, 0].detach().cpu().numpy()

    smoothed_final_left = patches2img(foj, foj.get_smoothed_patches())
    smoothed_final_right = patches2img(foj, foj.get_right_smoothed_patches())

    boundary_final_left = patches2img(foj, foj.get_boundary_patches()).squeeze()
    boundary_final_right = patches2img(foj, foj.get_right_boundary_patches()).squeeze()

    lrc_mask = get_LRC_mask(pred_disp_final, pred_disp_right, LRC_THRESHOLD)
    valid_gt = (gt_crop != 0)
    valid_mask = (boundary_final_left > BOUNDARY_THRESHOLD) & lrc_mask

    wind_EPE, wind_bad1, wind_bad3, wind_bad5, wind_mean_best_gt = compute_disparity_metrics_with_window(pred_disp_final, gt_for_window, window=WINDOW_SIZE, pred_valid=valid_mask)

    wind_bad1 *= 100
    wind_bad3 *= 100
    wind_bad5 *= 100

    print(f"Window Metrics: EPE: {wind_EPE:.4f}, Bad1: {wind_bad1:.2f}%, Bad3: {wind_bad3:.2f}%, Bad5: {wind_bad5:.2f}%, Mean Best GT: {wind_mean_best_gt:.4f}")

    left_img = foj.t_left_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    right_img = foj.t_right_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    final_disp_masked = pred_disp_final.copy().squeeze()
    final_disp_masked[~valid_mask] = np.nan

    vmin, vmax = gt_crop[gt_crop > 0].min(), gt_crop[gt_crop > 0].max()
    vmin = .9 * vmin
    vmax = 1.1 * vmax

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')
    gt_crop_masked = gt_crop.copy()
    gt_crop_masked[~(valid_gt & valid_mask)] = np.nan
    ax[0, 0].imshow(left_img)
    ax[0, 0].set_title("Left Image")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(right_img)
    ax[0, 1].set_title("Right Image")
    ax[0, 1].axis('off')

    ax[1, 0].imshow(smoothed_final_left)
    ax[1, 0].set_title("Smoothed Left")
    ax[1, 0].axis('off')

    ax[1, 1].imshow(smoothed_final_right)
    ax[1, 1].set_title("Smoothed Right")
    ax[1, 1].axis('off')

    ax[0, 2].imshow(boundary_final_left, cmap="gray")
    ax[0, 2].set_title("Boundary Left")
    ax[0, 2].axis('off')

    ax[1, 2].imshow(boundary_final_right, cmap="gray")
    ax[1, 2].set_title("Boundary Right")
    ax[1, 2].axis('off')

    ax[2, 0].imshow(gt_crop_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2, 0].set_title("GT Disparity (bnd)")
    ax[2, 0].axis('off')

    ax[2, 1].imshow(final_disp_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2, 1].set_title(f"Pred - Window EPE: {wind_EPE:.2f}, bad1: {wind_bad1:.2f}, bad3: {wind_bad3:.2f}")
    ax[2, 1].axis('off')

    ax[2, 2].imshow(gt_crop, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2, 2].set_title("GT Disparity")
    ax[2, 2].axis('off')

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    valid_percent = np.mean(valid_mask) * 100
    print(f"Valid percent: {valid_percent:.2f}%")

    return valid_percent, wind_EPE, wind_bad1, wind_bad3, wind_bad5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=19)
    parser.add_argument("--instereo_gt_dir", type=str, help="Directory containing the InStereo2K gt disparity maps")
    parser.add_argument("--results_dir", type=str, help="Directory to put processed .pth results")
    parser.add_argument("--viz_only", action="store_true", help="Visualize processed results")
    args = parser.parse_args()
    WINDOW_SIZE = args.window_size

    SAVE_BASE_DIR = args.results_dir
    IMAGE_BASE_DIR = args.instereo_gt_dir
    SAVE_VIS_DIR = os.path.join(args.results_dir, "vis")

    os.makedirs(SAVE_VIS_DIR, exist_ok=True)
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)

    VISUALIZE_ONLY = args.viz_only

    opts = create_opts(
        num_init_iters=30,
        num_refine_iters=1000,
        patch_size=40,
        stride=8
    )
    opts.parallel_mode = True

    if not VISUALIZE_ONLY:
        for index, (im0_path, im1_path, disp0_path) in tqdm(enumerate(get_all_instereo_paths(IMAGE_BASE_DIR))):
            foj, gt_crop = process_one_instereo_pair(im0_path, im1_path, disp0_path, opts)
            torch.save(foj, os.path.join(SAVE_BASE_DIR, f'foj_{index}.pth'))
            del foj
            del gt_crop
            torch.cuda.empty_cache()

    else:
        total_valid_percent = 0
        total_num = 0

        total_wind_EPE = 0
        total_wind_bad1 = 0
        total_wind_bad3 = 0
        total_wind_bad5 = 0
        for index, (im0_path, im1_path, disp0_path) in tqdm(enumerate(get_all_instereo_paths(IMAGE_BASE_DIR))):
            foj_path = os.path.join(SAVE_BASE_DIR, f'foj_{index}.pth')
            if not os.path.exists(foj_path):
                continue
            foj = torch.load(foj_path, weights_only=False)
            valid_percent, wind_EPE, wind_bad1, wind_bad3, wind_bad5 = visualize(foj, opts, disp0_path, os.path.join(SAVE_VIS_DIR, f'foj_{index}.png'))
            total_valid_percent += valid_percent
            total_num += 1

            total_wind_EPE += wind_EPE
            total_wind_bad1 += wind_bad1
            total_wind_bad3 += wind_bad3
            total_wind_bad5 += wind_bad5

        total_mean_valid_percent = total_valid_percent / total_num

        total_mean_wind_EPE = total_wind_EPE / total_num
        total_mean_wind_bad1 = total_wind_bad1 / total_num
        total_mean_wind_bad3 = total_wind_bad3 / total_num
        total_mean_wind_bad5 = total_wind_bad5 / total_num

        if total_num > 0:
            with open(os.path.join(SAVE_BASE_DIR, "results.txt"), "w") as f:
                f.write(f"Total Window EPE: {total_mean_wind_EPE:.4f}\n")
                f.write(f"Total Window Bad1: {total_mean_wind_bad1:.2f}%\n")
                f.write(f"Total Window Bad3: {total_mean_wind_bad3:.2f}%\n")
                f.write(f"Total Window Bad5: {total_mean_wind_bad5:.2f}%\n")
                f.write(f"Total Valid Percent: {total_mean_valid_percent:.2f}%\n")

            print(f"\nFinal Results ({total_num} images):")
            print(f"Total Window EPE: {total_mean_wind_EPE:.4f}")
            print(f"Total Window Bad1: {total_mean_wind_bad1:.2f}%")
            print(f"Total Window Bad3: {total_mean_wind_bad3:.2f}%")
            print(f"Total Window Bad5: {total_mean_wind_bad5:.2f}%")
            print(f"Total Valid Percent: {total_mean_valid_percent:.2f}%")
        else:
            print("No images were processed successfully.")