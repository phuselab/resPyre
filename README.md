# ResPyre - Respiratory Rate Estimation from Video

This repository contains code for estimating respiratory rate from video using different methods and datasets.

## Overview

The main script [run_all.py](run_all.py) supports:

1. Extracting respiratory signals from videos using different methods
2. Evaluating the results with multiple metrics 
3. Printing the evaluation metrics

## Usage

```bash
python run_all.py -a <action> -d <results_dir>
```

Arguments:
- `-a`: Action to perform 
  - `0`: Extract respiratory signals
  - `1`: Evaluate results
  - `2`: Print metrics
- `-d`: Directory to save/load results (default: `results/`)

## Supported Methods

The following methods are implemented:

1. Deep Learning Methods:
- [`MTTS_CAN`](deep/MTTS_CAN/train.py): Multi-Task Temporal Shift Attention Network 
- [`BigSmall`](deep/BigSmall/predict_vitals.py): BigSmall Network

2. Motion-based Methods:
- [`OF_Deep`](run_all.py): Deep Optical Flow estimation
- [`OF_Model`](run_all.py): Traditional Optical Flow (Farneback)
- [`DoF`](run_all.py): Difference of Frames
- [`profile1D`](run_all.py): 1D Motion Profile

3. rPPG-based Methods:
- [`peak`](run_all.py): Peak detection
- [`morph`](run_all.py): Morphological analysis
- [`bss_ssa`](run_all.py): Blind Source Separation with SSA
- [`bss_emd`](run_all.py): Blind Source Separation with EMD

## Supported Datasets 

The code works with the following datasets:

- [`BP4D`](run_all.py): BP4D Dataset
- [`COHFACE`](run_all.py): COHFACE Dataset  
- [`MAHNOB`](run_all.py): MAHNOB-HCI Dataset

## Example Usage

1. Extract respiratory signals using deep learning methods:

```python
methods = [BigSmall(), MTTS_CAN()]
datasets = [BP4D(), COHFACE()]
extract_respiration(datasets, methods, "results/")
```

2. Evaluate the results:

```bash
python run_all.py -a 1 -d results/
```

3. Print metrics:

```bash 
python run_all.py -a 2 -d results/
```

## Requirements

Required packages are listed in [requirements.txt](requirements.txt). Key dependencies include:

- TensorFlow 2.2-2.4
- OpenCV
- SciPy
- NumPy
- Matplotlib

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) file for details.