"""
Script to visualize preprocessed foj stereo images from middlebury
using saved npz (disparity_map, boundaries, smoothed_left = processor.process_big_stereo(verbose=True))
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stereo_BIG_BLOCKS_SAVE_RIGHT import get_LRC_mask, read_pfm
from metrics_utils import compute_disparity_metrics_with_window
import argparse

BOUNDARY_THRESHOLD = 0.1
LRC_THRESHOLD = 1
WINDOW_SIZE = 19

def visualize_pair(pair_name, results_dir="middlebury_results"):
    """
    Visualize a single processed pair
    """
    results_path = os.path.join(results_dir, f"{pair_name}.npz")

    os.makedirs(os.path.join(results_dir, "visualization"), exist_ok=True)
    
    data = np.load(results_path, allow_pickle=True)
    
    disparity_map = data['disparity_map']
    disparity_map_right = data['disparity_map_right']
    boundaries = data['boundaries']
    smoothed_left = data['smoothed_left']
    left_img = data['left_img']
    right_img = data['right_img']

    right_boundaries = data['right_boundaries']
    smoothed_right = data['smoothed_right']
    
    opts = data['opts'].item() if isinstance(data['opts'], np.ndarray) else data['opts']

    disparity_map *= opts['R'] / 2 # scale to pixels [-1,1] -> [-R/2, R/2]
    disparity_map_right *= opts['R'] / 2

    # mask by LRC and boundaries
    boundary_mask = (boundaries > BOUNDARY_THRESHOLD)
    LRC_mask = get_LRC_mask(disparity_map, disparity_map_right, LRC_THRESHOLD)
    valid_mask = boundary_mask & LRC_mask

    gt_disparity = read_pfm(f"middlebury2014/{pair_name}/disp0.pfm")
    scale_factor = disparity_map.shape[0] / gt_disparity.shape[0]
    gt_disparity = gt_disparity * scale_factor
    gt_disparity[np.isinf(gt_disparity)] = 0
    gt_disparity = cv2.resize(gt_disparity, (disparity_map.shape[1], disparity_map.shape[0]), interpolation=cv2.INTER_AREA)

    # save raw gt disp for the window metric
    gt_for_window = gt_disparity.copy()
    gt_for_window[(gt_for_window == 0)] = np.nan
    # NOTE - using valid_mask since window fn handles nans
    wind_EPE, wind_bad1, wind_bad3, wind_bad5, wind_mean_best_gt = compute_disparity_metrics_with_window(disparity_map, gt_for_window, window=WINDOW_SIZE, pred_valid=valid_mask)

    wind_bad1 *= 100
    wind_bad3 *= 100
    wind_bad5 *= 100

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    
    # Row 0: Left image, Smoothed left, Left boundaries
    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(smoothed_left)
    axes[0, 1].set_title('Smoothed Left')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(boundaries, cmap='gray')
    axes[0, 2].set_title('Left Boundaries')
    axes[0, 2].axis('off')
    
    # Row 1: Right image, Smoothed right, Right boundaries
    axes[1, 0].imshow(right_img)
    axes[1, 0].set_title('Right Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(smoothed_right)
    axes[1, 1].set_title('Smoothed Right')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(right_boundaries, cmap='gray')
    axes[1, 2].set_title('Right Boundaries')
    axes[1, 2].axis('off')
    
    # Row 2: GT disparity, pred disp masked, pred disp dense lrc masked
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')
    vmin, vmax = gt_disparity[np.isfinite(gt_disparity)].min(), gt_disparity[np.isfinite(gt_disparity)].max()
    axes[2, 0].imshow(gt_disparity, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 0].set_title('GT Disparity Full')
    axes[2, 0].axis('off')

    disparity_map_masked = disparity_map.copy()
    disparity_map_masked[~valid_mask] = np.nan
    axes[2, 1].imshow(disparity_map_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 1].set_title(f'Final WINDOW EPE: {wind_EPE:.3f} px, bad1: {wind_bad1:.2f}%, bad3: {wind_bad3:.2f}%')
    axes[2, 1].axis('off')

    disparity_map_dense_masked = disparity_map.copy()
    disparity_map_dense_masked[~boundary_mask] = np.nan
    axes[2, 2].imshow(disparity_map_dense_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 2].set_title('Disp Boundary Only Masked')
    axes[2, 2].axis('off')


    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"visualization/{pair_name}.png"))

    plt.close()

    print(f"WINDOW EPE: {wind_EPE:.2f} px, WINDOW BAD-1: {wind_bad1:.2f}%, WINDOW BAD-3: {wind_bad3:.2f}%, WINDOW BAD-5: {wind_bad5:.2f}%")

    valid_percent = np.mean(valid_mask) * 100
    
    return valid_percent, wind_EPE, wind_bad1, wind_bad3, wind_bad5, wind_mean_best_gt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing the processed .npz results to evaluate")
    args = parser.parse_args()
    results_dir = args.results_dir

    all_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.npz') and f != 'summary.npz'])

    print(f"Found {len(all_files)} processed pairs")
    
    total_valid_percent = 0    

    total_wind_EPE = 0
    total_wind_bad1 = 0
    total_wind_bad3 = 0
    total_wind_bad5 = 0
    total_wind_mean_best_gt = 0

    csv_rows = []
    
    for filename in all_files:
        try:
            pair_name = filename.replace('.npz', '')
            valid_percent, wind_EPE, wind_bad1, wind_bad3, wind_bad5, wind_mean_best_gt = visualize_pair(pair_name, results_dir)

            total_valid_percent += valid_percent

            total_wind_EPE += wind_EPE
            total_wind_bad1 += wind_bad1
            total_wind_bad3 += wind_bad3
            total_wind_bad5 += wind_bad5
            total_wind_mean_best_gt += wind_mean_best_gt

            csv_rows.append([pair_name, wind_EPE, wind_bad1, wind_bad3, wind_bad5, wind_mean_best_gt])
        except Exception as e:
            print(f"Failed {filename} {e}")
            exit()
    
    # save metrics
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'wind_EPE', 'wind_bad1', 'wind_bad3', 'wind_bad5', 'wind_mean_best_gt'])
        writer.writerows(csv_rows)
    
    print(f"Average WINDOW EPE: {total_wind_EPE / len(all_files):.4f}")
    print(f"Average WINDOW BAD-1: {total_wind_bad1 / len(all_files):.2f}%")
    print(f"Average WINDOW BAD-3: {total_wind_bad3 / len(all_files):.2f}%")
    print(f"Average WINDOW BAD-5: {total_wind_bad5 / len(all_files):.2f}%")
    print(f"Average WINDOW MEAN BEST GT: {total_wind_mean_best_gt / len(all_files):.2f}")

    print(f"Average Valid Percent: {total_valid_percent / len(all_files):.2f}%")




if __name__ == "__main__":
    main()
