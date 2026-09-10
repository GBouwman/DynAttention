# DynVision Data Preparation & Conversion Guide

This document outlines the exact end-to-end pipeline DynVision uses to download datasets, organize them into manageable groups, and convert them into highly-optimized FFCV `.beton` files for training. 

The entire process is orchestrated through Snakemake rules defined in [`snake_data.smk`](file:///c:/Users/Glenn/OneDrive%20-%20Radboud%20Universiteit/3.PFCmodels/dynvision/dynvision/workflow/snake_data.smk) and configured via `config_data.yaml`.

> [!NOTE]
> All snakemake commands should be executed from the `dynvision/workflow` directory. Paths should be customized to your specific environment or left to the wildcard defaults.

---

## The 3-Step Pipeline

DynVision splits data management into three distinct layers:
1. **Raw Storage** (`data/raw/`)
2. **Interim Organization** (`data/interim/`)
3. **Processed/FFCV Conversion** (`data/processed/`)

### Step 1: Downloading & Raw Storage
The first step is retrieving the canonical dataset (e.g., CIFAR10, CIFAR100, MNIST). This is handled by the `get_data` rule which triggers a Python download script.

**Command:**
```bash
snakemake <project_paths.data.raw>/<dataset_name>/<subset>/
```
*Example: `snakemake /home/glenn/DynVision_Working/data/raw/cifar10/train/`*

**What happens:**
The raw images are downloaded in their original format and placed strictly in `data/raw/cifar10/train/`. This folder remains completely untouched after this step.

### Step 2: Interim Organization & Grouping
Because experiments often require evaluating on specific subsets of classes (e.g., "invertebrates" only), DynVision creates symlinks rather than duplicating image files. 

**Command:**
```bash
snakemake <project_paths.data.interim>/<dataset_name>/<subset>_<group>.ready
```
*Example: `snakemake /home/glenn/DynVision_Working/data/interim/cifar10/train_all.ready`*

**What happens:**
1. The `symlink_data_subsets` rule creates canonical symlinks pointing back to the raw files.
2. The `symlink_data_groups` rule filters these symlinks based on the `data_groups` defined in `config_data.yaml` (if the group is `all`, it includes all classes).
3. A `.ready` flag file is created to indicate downstream steps can proceed.

> [!TIP]
> **Testing occurs at this stage!** When you evaluate models on test data, the standard PyTorch DataLoaders (e.g., `StimulusDurationDataLoader`) read directly from these `data/interim/` symlinked directories. This allows for complex on-the-fly experimental manipulations that FFCV cannot easily handle.

### Step 3: FFCV `.beton` Conversion
For training, PyTorch DataLoaders are too slow. We must convert the organized interim training data into compiled FFCV `.beton` files.

**Command:**
```bash
snakemake <project_paths.data.processed>/<dataset_name>/train_all/train.beton
```
*Example: `snakemake /home/glenn/DynVision_Working/data/processed/cifar10/train_all/train.beton`*

**What happens:**
The `build_ffcv_datasets` rule reads the `train_all` directory from the interim step and does two things simultaneously:
1. Compiles the dataset into a high-performance FFCV `.beton` file.
2. Applies a `train_ratio` (defined in the configuration) to automatically split out a validation set. 

**Resulting Output:**
```
data/processed/<dataset_name>/train_all/
├── train.beton
└── val.beton
```

> [!IMPORTANT]  
> Notice that the validation file (`val.beton`) is created directly alongside the `train.beton` inside the `train_all` directory. There is no separate `test_all/val.beton` target.

---

## Summary of Usage Paths
Once you have run the three steps above, your system is fully prepared:

* **For Fast Training**: Models use `get_ffcv_dataloader()` pointing to `data/processed/cifar10/train_all/train.beton` and `val.beton`.
* **For Experimental Testing**: Models use custom PyTorch dataloaders (like `StimulusContrastLoader`) pointing to `data/interim/cifar10/test_all/` or specific testing groups like `test_invertebrates/`.
