# Bilateral Symmetry-based Augmentation Method for Improved Tooth Segmentation in Panoramic X-rays

This project focuses on enhancing tooth segmentation in panoramic X-rays using a novel bilateral symmetry-based augmentation technique. The steps below outline the preparation, augmentation, and comprehensive analysis processes.

## Prepare Segmentation Dataset

Run the `process...` functions in `process_dataset.py` to convert the dataset into quadrant and 32-class segmentation masks for segmentation models. The quadrant segmentation dataset can be generated from the original quadrant dataset, and the 32-class segmentation dataset can be generated from the original quadrant_enumeration dataset.

## Bilateral Symmetry-based Augmentation

We propose a novel method to expand the dataset to four times its original size.

### Steps

1. **Train SE U-Net for Quadrant Segmentation**

    ```sh
    python train_unet.py \
        --output_dir outputs/output_unet_quadrant_16 \
        --dataset_dir dentex_dataset/segmentation/quadrant \
        --num_classes 4 --model seunet --batch_size 16
    ```

2. **Generate Augmented Dataset**

    Run `data_augmentation_all.py` to create a new dataset in `dentex_dataset/segmentation/enumeration32_augmentation_all`. This new dataset will have four times the images and masks compared to `dentex_dataset/segmentation/enumeration32`.

3. **Train U-Net, SE U-Net, and TransUNet on the Augmented Dataset**

## Comprehensive Analysis

1. **Split the Dataset**

    Run `split_train_val_test.py` to split the enumeration32 segmentation dataset into train (380), validation (127), and test (127) sets. The split dataset will be saved in `dentex_dataset/segmentation/enumeration32_train_val_test`.

2. **Generate Training Sets**

    Split the train set into subsets starting from a random 80 images, adding 50 randomly to create the next set by running `split_train.py` in the `bilateral_symmetry_augmentation` folder. This will produce `train_80`, `train_130`, `train_180`, `train_230`, `train_280`, `train_330`, and `train_380`. These sets will be saved in the same directory.

3. **Generate Augmented Data for Each Set**

    Run `data_augmentation_train_range.py` to generate augmented data for each of these training sets. The augmented data will be saved in `dentex_dataset/segmentation/enumeration32_augmentation`.

4. **Train and Test Segmentation Models on Each Set**

    Run `training_32_{model}.sh` and `testing_32_{model}.sh` in the `train_test` folder to train U-Net, SE U-Net, and TransUNet on each of these sets and test on their best checkpoints.
