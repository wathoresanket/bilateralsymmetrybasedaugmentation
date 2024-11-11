import os
import json
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import elasticdeform

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        # load image names from json file
        with open(os.path.join(dataset_dir, "image_names.json"), "r") as f:
            self.image_names = json.load(f)

    def __getitem__(self, index) -> tuple[Image.Image, Image.Image]:
        image_name = self.image_names[index]
        
        # open and convert image to grayscale
        with Image.open(os.path.join(self.dataset_dir, "xrays", image_name)) as image:
            image = image.convert("L").copy()
        
        # open mask image
        with Image.open(os.path.join(self.dataset_dir, "masks", image_name)) as mask:
            mask = mask.copy()
        
        return image_name, image, mask

    def __len__(self) -> int:
        return len(self.image_names)


class Preload(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # preload all data into memory
        self.data = [dataset[i] for i in range(len(dataset))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.dataset)


def random_rotate(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    # rotate image and mask by a random angle
    angle = random.uniform(-30, 30)
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    rotated_mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)
    return rotated_image, rotated_mask


def apply_elastic_deformation(image: Image.Image, mask: Image.Image):
    image_np = np.array(image)
    mask_np = np.array(mask)
    # get coordinates of non-zero mask pixels
    non_zero_coords = np.where(mask_np > 0)
    if non_zero_coords[0].size > 0:
        min_row, max_row = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
        min_col, max_col = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
        # define crop region based on mask
        crop_region = (slice(min_row, max_row + 1), slice(min_col, max_col + 1))
        image_deformed, mask_deformed = elasticdeform.deform_random_grid(
            [image_np, mask_np], crop=crop_region, order=[3, 0], sigma=40, points=7
        )
    else:
        image_deformed, mask_deformed = image_np, mask_np
    image_deformed = Image.fromarray(image_deformed)
    mask_deformed = Image.fromarray(mask_deformed)
    return image_deformed, mask_deformed


def save_deformed_images(output_dir, image_name, original_image, original_mask, deformed_image, deformed_mask):
    image_output_dir = os.path.join(output_dir, "xrays")
    mask_output_dir = os.path.join(output_dir, "masks")
    # create output directories if they don't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    # save original and deformed images and masks
    original_image.save(os.path.join(image_output_dir, f"{image_name}"))
    original_mask.save(os.path.join(mask_output_dir, f"{image_name}"))
    deformed_image.save(os.path.join(image_output_dir, f"deformed_{image_name}"))
    deformed_mask.save(os.path.join(mask_output_dir, f"deformed_{image_name}"))


# example usage for multiple directories:
directory_suffixes = [80, 130, 180, 230, 280, 330, 380]

for suffix in directory_suffixes:
    train_dataset_dir = f"dentex_dataset/segmentation/enumeration32_train_val_test/train_{suffix}/"
    output_dir = f"dentex_dataset/segmentation/enumeration32_elastic_grid_based_augmentation_train_range/train_{suffix}/"
    
    # load image names from json file
    with open(os.path.join(train_dataset_dir, "image_names.json"), "r") as f:
        image_names = json.load(f)
    
    train_dataset = Preload(SegmentationDataset(train_dataset_dir))
    new_image_names = []

    for i in range(len(train_dataset)):
        image_name, image, mask = train_dataset[i]
        rotated_image, rotated_mask = random_rotate(image, mask)
        image_deformed, mask_deformed = apply_elastic_deformation(rotated_image, rotated_mask)
        save_deformed_images(output_dir, image_name, image, mask, image_deformed, mask_deformed)
        new_image_names.append(f"deformed_{image_name}")

    updated_image_names = image_names + new_image_names
    # save updated image names to json file
    with open(os.path.join(output_dir, "image_names.json"), "w") as f:
        json.dump(updated_image_names, f)

    print(f"deformation and saving complete for directory train_{suffix}.")

print("Done")
