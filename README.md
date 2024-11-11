# Bilateral Symmetry-based Augmentation for Improved Tooth Segmentation in Panoramic X-rays

This repository contains the code for a bilateral symmetry-based augmentation technique aimed at enhancing tooth segmentation accuracy in panoramic X-rays. Leveraging the inherent symmetry in these images, our method significantly expands the dataset size and improves the performance of deep learning models, including U-Net, SE U-Net, and TransUNet.

Key improvements include:
- An 8% increase in the Dice Similarity Coefficient (DSC), achieving 76.7% for TransUNet.
- Superior performance over rigid transform-based and elastic grid-based augmentations, with an additional 5% DSC improvement on average.

---

## Workflow

### 1. Prepare Segmentation Dataset
Use `process_dataset.py` to generate:
- **Quadrant Segmentation Masks** from the original quadrant dataset.
- **32-Class Segmentation Masks** from the quadrant_enumeration dataset.

### 2. Bilateral Symmetry-based Augmentation
Expand the dataset fourfold by running `bilateral_symmetry_based_augmentation` to generate data in `dentex_dataset/segmentation/enumeration32_bilateral_symmetry_based_augmentation`.

### 3. Model Training on Augmented Dataset
Train U-Net, SE U-Net, and TransUNet models on the augmented dataset.

---

## Comprehensive Analysis

1. **Dataset Splitting:**
   Use `split_train_val_test.py` to divide the enumeration32 dataset into training (380), validation (127), and testing (127) sets, saved in `dentex_dataset/segmentation/enumeration32_train_val_test`.

2. **Generate Training Subsets:**
   Run `split_train.py` to create training subsets (`train_80` to `train_380`) by incrementally adding images. These subsets are saved in the same directory.

3. **Augmentation Comparisons:**
   Create augmented datasets for each training subset with:
   - **Bilateral Symmetry-based Augmentation:** `bilateral_symmetry_based_augmentation_train_range.py`
   - **Rigid Transform-based Augmentation:** `rigid_transform_based_augmentation_train_range.py`
   - **Elastic Grid-based Augmentation:** `elastic_grid_based_augmentation_train_range.py`

4. **Train and Test Models:**
   Use `training_32_{model}.sh` and `testing_32_{model}.sh` scripts to train and evaluate U-Net, SE U-Net, and TransUNet models on the datasets with four augmentation methods:
   - No augmentation
   - Rigid transform-based augmentation
   - Elastic deformation-based augmentation
   - Bilateral symmetry-based augmentation
