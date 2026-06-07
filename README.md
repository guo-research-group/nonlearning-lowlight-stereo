# Usage

## Configure the environment

### Installation
```bash
git clone https://github.com/guo-research-group/nonlearning-lowlight-stereo.git
cd nonlearning-lowlight-stereo
conda create -n nonlearning_stereo python=3.13.5
conda activate nonlearning_stereo
pip install -r requirements.txt
```

The original and noisy datasets used for development and evaluation can be downloaded [here](https://purdue0-my.sharepoint.com/:f:/g/personal/wangjx_purdue_edu/IgANEeQG8iy1QbMa51cm2tbRARBChoVteyIyBc5zcYNoufI?e=HZqazY)

## Estimation

To estimate disparity from scratch, download and extract the datasets so that the repository root contains the following directories:
```
middlebury2014noisy_alpha2
InStereo2kNoisy
instereo_test_set_noisy
```

Estimate disparity on the noisy Middlebury dataset. Results are saved as `.npz` files for later evaluation:
```bash
python EVAL_BIG_MIDDLEBURY.py --output_dir "middlebury_results"
```

Estimate disparity on the noisy InStereo2K train and test sets. Results are saved as `.pth` files for later evaluation:
```bash
python EVAL_INSTEREO.py --instereo_gt_dir "InStereo2kNoisy" --results_dir "instereo_results"
python EVAL_INSTEREO.py --instereo_gt_dir "instereo_test_set_noisy" --results_dir "instereo_results_test_set"
```

## Evaluation

To reproduce the metrics reported in the paper, download and extract the datasets and preprocessed results so that the repository root contains the following directories:
```
middlebury2014
InStereo2kNoisy
instereo_test_set_noisy
middlebury_results
instereo_results
instereo_results_test_set
```

Compute the metrics for the estimated Middlebury disparities:
```bash
python VISUALIZE_BIG_ALL.py --results_dir "middlebury_results"
```

Compute the metrics for the estimated InStereo2K disparities (the `--viz_only` flag evaluates the preprocessed results instead of recomputing them):
```bash
python EVAL_INSTEREO.py --instereo_gt_dir "InStereo2kNoisy" --results_dir "instereo_results" --viz_only
python EVAL_INSTEREO.py --instereo_gt_dir "instereo_test_set_noisy" --results_dir "instereo_results_test_set" --viz_only
```