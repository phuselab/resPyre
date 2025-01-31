# ResPyre - Respiratory Rate Estimation from Video (resPyre)

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

## Extending the Code

### Adding New Datasets

To add a new dataset, create a class that inherits from `DatasetBase` and implement the required methods:

```python
class NewDataset(DatasetBase):
    def __init__(self):
        super().__init__()
        self.name = 'new_dataset'  # Unique dataset identifier
        self.path = self.data_dir + 'path/to/dataset/'
        self.fs_gt = 1000  # Ground truth sampling frequency
        self.data = []

    def load_dataset(self):
        # Load dataset metadata and populate self.data list
        # Each item should be a dict with:
        # - video_path: path to video file
        # - subject: subject ID
        # - chest_rois: [] (empty list, populated during processing)
        # - face_rois: [] (empty list, populated during processing) 
        # - rppg_obj: [] (empty list, populated during processing)
        # - gt: ground truth respiratory signal

    def load_gt(self, trial_path):
        # Load ground truth respiratory signal for a trial
        pass

    def extract_ROI(self, video_path, region='chest'):
        # Extract ROIs from video for given region ('chest' or 'face')
        pass

    def extract_rppg(self, video_path, method='cpu_CHROM'):
        # Extract rPPG signal from video
        pass
```

### Adding New Methods

To add a new respiratory rate estimation method, inherit from `MethodBase`:

```python
class NewMethod(MethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'new_method'  # Unique method identifier
        self.data_type = 'chest'  # Input type: 'chest', 'face' or 'rppg'

    def process(self, data):
        # Implement respiratory signal extraction
        # data contains:
        # - chest_rois: list of chest ROI frames 
        # - face_rois: list of face ROI frames
        # - rppg_obj: rPPG signal object
        # - fps: video framerate
        # Return the extracted respiratory signal
        pass
```

After implementing the new classes, you can use them with the existing pipeline:

```python
methods = [NewMethod()]
datasets = [NewDataset()]
extract_respiration(datasets, methods, "results/")
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