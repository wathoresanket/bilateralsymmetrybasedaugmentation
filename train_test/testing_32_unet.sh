#!/bin/bash
cd ..

initial_subset_size=80
increment=50
max_training_size=380

# no augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    output_dir="unet_test/output_unet_enum32_16_no_augmentation_$size"
    model_path=$(ls -t unet_train/output_unet_enum32_16_no_augmentation_$size/*.pth | head -1) 
    python test_unet.py \
        --output_dir "$output_dir" \
        --dataset_dir dentex_dataset/segmentation/enumeration32_train_val_test/test \
        --model_path "$model_path" \
        --num_classes 32 --model unet --batch_size 16
done

# random augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    output_dir="unet_test/output_unet_enum32_16_random_augmentation_$size"
    model_path=$(ls -t unet_train/output_unet_enum32_16_random_augmentation_$size/*.pth | head -1) 
    python test_unet.py \
        --output_dir "$output_dir" \
        --dataset_dir dentex_dataset/segmentation/enumeration32_train_val_test/test \
        --model_path "$model_path" \
        --num_classes 32 --model unet --batch_size 16
done

# bilateral symmetry-based augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    output_dir="unet_test/output_unet_enum32_16_bilateral_symmetry_based_augmentation_$size"
    model_path=$(ls -t unet_train/output_unet_enum32_16_bilateral_symmetry_based_augmentation_$size/*.pth | head -1) 
    python test_unet.py \
        --output_dir "$output_dir" \
        --dataset_dir dentex_dataset/segmentation/enumeration32_train_val_test/test \
        --model_path "$model_path" \
        --num_classes 32 --model unet --batch_size 16
done