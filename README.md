# BraTS2024_BioMedIAMBZ

## How To Use

### Installation

``` bash
conda create -n brats python=3.8
pip install -r requirements.txt
```

### Data Preprocessing

1. Download the BraTS2023 Adult Glioma dataset and put it on the `dataset/` folder, so it will contain the following:
```
├── dataset
│   ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
│   ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
│   ├── brats21_folds.json
│   ├── BraTS2023_2017_GLI_Mapping.xlsx
```
2. Run `preprocessing.py`, but please check the `source_directory` and `target_directory` variables to make sure everything is correct.
```
conda activate brats-gli
python preprocessing.py 
```

### Training
For training a `MedNeXt` model, you can run the following command (but you may need to configure your wandb account beforehand):
```
python mednext_train.py
```

### 5-Fold CV Dice & HD95
To calculate 5-fold CV Dice and HD95, we need to do two things;
- get predictions from 5-fold (`cv-get-predictions.py`).
- run post-processing and evaluation (`cv-postprocessing-and-eval.py`).
- The idea of separating `cv-get-predictions.py` and `cv-postprocessing-and-eval.py` is to allow us to tune post-processing faster.