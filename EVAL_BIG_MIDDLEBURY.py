"""
Runs stereo foj on middlebury using big blocks and saves results to npz for later review
"""

from stereo_BIG_BLOCKS_SAVE_RIGHT import create_big_opts, StereoBigProcessor, read_pfm
import os
import numpy as np
from tqdm import tqdm


def get_middlebury_pairs(dataset_dir="middlebury2021"):
    """
    Returns list of stereo pair dirs
    """
    pairs = sorted([d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d))])
    return pairs


def process_single_pair(pair_dir, opts, output_dir, verbose=True):
    """
    Process a single Middlebury stereo pair and save results to npz file
    Returns dict with processing results or None if failed
    """
    left_path = os.path.join(pair_dir, "im0.png")
    right_path = os.path.join(pair_dir, "im1.png")
    gt_disp_path = os.path.join(pair_dir, "disp0.pfm")
    
    if not os.path.exists(left_path):
        print(f"Missing left image: {left_path}")
        return None
    if not os.path.exists(right_path):
        print(f"Missing right image: {right_path}")
        return None
    
    has_gt = os.path.exists(gt_disp_path)
    
    try:
        processor = StereoBigProcessor(
            left_path, right_path, opts, 
            gt_disparity_path=gt_disp_path if has_gt else None
        )
        disparity_map, boundaries, smoothed_left, right_boundaries, smoothed_right, \
        disparity_map_p1, boundaries_p1, smoothed_left_p1, right_boundaries_p1, smoothed_right_p1, disparity_map_right = processor.process_big_stereo(verbose=verbose)
        
        gt_disparity = None
        if has_gt:
            gt_disparity = read_pfm(gt_disp_path)
        
        results = {
            'disparity_map': disparity_map,
            'disparity_map_right': disparity_map_right,
            'boundaries': boundaries,
            'smoothed_left': smoothed_left,
            'right_boundaries': right_boundaries,
            'smoothed_right': smoothed_right,

            'disparity_map_pass1': disparity_map_p1,
            'boundaries_pass1': boundaries_p1,
            'smoothed_left_pass1': smoothed_left_p1,
            'right_boundaries_pass1': right_boundaries_p1,
            'smoothed_right_pass1': smoothed_right_p1,

            'left_img': processor.left_img,
            'right_img': processor.right_img,
            'gt_disparity': gt_disparity,
            'H': processor.H,
            'W': processor.W,


            'opts': {
                'R': opts.R,
                'stride': opts.stride,
                'block_size': opts.block_size,
                'n_margin_patch': opts.n_margin_patch,
                'num_init_iters': opts.num_initialization_iters,
                'num_refine_iters': opts.num_refinement_iters,
                'lambda_stereo_final': opts.lambda_stereo_final,
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="middlebury2014noisy_alpha2")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--block_size', type=int, default=300)
    parser.add_argument('--n_margin_patch', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=40)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--num_init_iters', type=int, default=30)
    parser.add_argument('--num_refine_iters', type=int, default=1000)
    parser.add_argument('--sgm_P1', type=int, default=20)
    parser.add_argument('--sgm_P2', type=int, default=200)
    parser.add_argument('--sgm_alpha_boundary', type=float, default=2.0)
    args = parser.parse_args()


    DATASET_DIR = args.dataset_dir
    OUTPUT_DIR = args.output_dir
    BLOCK_SIZE = args.block_size
    N_MARGIN = args.n_margin_patch
    PATCH_SIZE = args.patch_size
    STRIDE = args.stride
    NUM_INIT_ITERS = args.num_init_iters
    NUM_REFINE_ITERS = args.num_refine_iters
    LAMBDA_STEREO = 10
    SGM_P1 = args.sgm_P1
    SGM_P2 = args.sgm_P2
    SGM_ALPHA_BOUNDARY = args.sgm_alpha_boundary
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pairs = get_middlebury_pairs(DATASET_DIR)
    print(f"Found {len(pairs)} Middlebury stereo pairs")
    
    opts = create_big_opts(
        block_size=(BLOCK_SIZE, BLOCK_SIZE),
        n_margin_patch=N_MARGIN,
        num_init_iters=NUM_INIT_ITERS,
        num_refine_iters=NUM_REFINE_ITERS,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        lambda_stereo_final=LAMBDA_STEREO,
        sgm_P1=SGM_P1,
        sgm_P2=SGM_P2,
        sgm_alpha_boundary=SGM_ALPHA_BOUNDARY
    )
    
    print(f"\nConfiguration:")
    print(f"  Block size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    print(f"  Margin patches: {N_MARGIN}")
    print(f"  Patch size (R): {PATCH_SIZE}")
    print(f"  Stride: {STRIDE}")
    print(f"  Init iters: {NUM_INIT_ITERS}")
    print(f"  Refine iters: {NUM_REFINE_ITERS}")
    print(f"  Lambda stereo: {LAMBDA_STEREO}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()
    
    results_summary = {}
    for pair_name in tqdm(pairs, desc="Processing pairs"):
        pair_dir = os.path.join(DATASET_DIR, pair_name)
        output_path = os.path.join(OUTPUT_DIR, f"{pair_name}.npz")
        
        print(f"Processing: {pair_name}")
        results = process_single_pair(pair_dir, opts, OUTPUT_DIR)
        
        if results is not None:
            np.savez_compressed(output_path, **results)
            print(f"  [SAVED] {output_path}")
            
            disp = results['disparity_map']
            results_summary[pair_name] = {
                'status': 'success',
                'disp_min': float(disp.min()),
                'disp_max': float(disp.max()),
                'disp_mean': float(disp.mean()),
                'shape': (results['H'], results['W']),
            }
        else:
            results_summary[pair_name] = {'status': 'failed'}
    
    summary_path = os.path.join(OUTPUT_DIR, "summary.npz")
    np.savez(summary_path, summary=results_summary)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
