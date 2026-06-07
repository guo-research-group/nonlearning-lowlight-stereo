import numpy as np

def compute_disparity_metrics_with_window(pred: np.ndarray, gt: np.ndarray, window: int = 19, pred_valid: np.ndarray = None) -> tuple:
    # Squeeze out any leading batch/channel dims → ensure shape is (H, W)
    pred = np.squeeze(pred).astype(np.float32)
    gt   = np.squeeze(gt).astype(np.float32)

    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    assert window % 2 == 1, "window size must be odd"
    if pred_valid is not None:
        assert pred_valid.shape == pred.shape

    H, W = pred.shape
    half = window // 2

    gt_padded = np.pad(gt, half, mode="edge")

    # set GT nans to inf so argmin doesnt return nan index
    gt_padded[~np.isfinite(gt_padded)] = np.inf

    from numpy.lib.stride_tricks import sliding_window_view
    gt_windows = sliding_window_view(gt_padded, (window, window))  # (H, W, window, window)

    abs_diff = np.abs(gt_windows - pred[:, :, None, None])

    flat_best_idx = np.argmin(abs_diff.reshape(H, W, -1), axis=2)

    best_abs_diff = abs_diff.reshape(H, W, -1)[
        np.arange(H)[:, None],
        np.arange(W)[None, :],
        flat_best_idx,
    ]

    best_gt = gt_windows.reshape(H, W, -1)[
        np.arange(H)[:, None],
        np.arange(W)[None, :],
        flat_best_idx,
    ]

    # Ignore pixels where GT is NaN
    if pred_valid is not None:
        valid = np.isfinite(best_abs_diff) & np.isfinite(best_gt) & pred_valid
    else:
        valid = np.isfinite(best_abs_diff) & np.isfinite(best_gt)
    
    eval_coverage = valid.sum() / (H*W)
    print(f"Eval coverage percentage: {eval_coverage * 100:.2f}%")

    EPE  = float(np.mean(best_abs_diff[valid]))
    bad1 = float(np.mean(best_abs_diff[valid] > 1.0))
    bad3 = float(np.mean(best_abs_diff[valid] > 3.0))
    bad5 = float(np.mean(best_abs_diff[valid] > 5.0))
    mean_best_gt = float(np.mean(best_gt[valid]))

    return EPE, bad1, bad3, bad5, mean_best_gt