#!/bin/bash
cd ..

initial_subset_size=80
increment=50
max_training_size=380

# no augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet_no_augmentation.py \
        --output_dir "transunet_train/output_unet_enum32_16_no_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done

# random augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet_random_augmentation.py \
        --output_dir "transunet_train/output_unet_enum32_16_random_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done

# bilateral symmetry-based augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_transunet_no_augmentation.py \
        --output_dir "transunet_train/output_unet_enum32_16_bilateral_symmetry_based_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_bilateral_symmetry_based_augmentation/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --batch_size 16
done