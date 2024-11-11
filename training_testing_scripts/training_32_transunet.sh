#!/bin/bash
cd ..

initial_subset_size=80
increment=50
max_training_size=380

# no augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet.py \
        --output_dir "outputs/transunet/train/no_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done

# rigid transform-based augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet.py \
        --output_dir "outputs/transunet/train/rigid_transform_based_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_rigid_transform_based_augmentation_train_range/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done

# elastic grid-based augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet.py \
        --output_dir "outputs/transunet/train/elastic_grid_based_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_elastic_grid_based_augmentation_train_range/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done

# bilateral symmetry-based augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet.py \
        --output_dir "outputs/transunet/train/bilateral_symmetry_based_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_bilateral_symmetry_based_augmentation_train_range/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done