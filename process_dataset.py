import os
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def process_seg_quadrant():
    """
    draw segmentation masks for enumeration32
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant/train_quadrant.json")
    mkdirs("dentex_dataset/segmentation/quadrant/masks")
    mkdirs("dentex_dataset/segmentation/quadrant/xrays")

    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }  # remap because category names are different between quadrant and quadrant_enumeration/quadrant_enumeration_disease
    
    image_names = []
    for image_info in dataset_json["images"]:
        image_names.append(image_info["file_name"])
        # draw mask for each image
        image = Image.open(f"dentex_dataset/origin/quadrant/xrays/{image_info['file_name']}")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for annotation in dataset_json["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                points = np.array(annotation["segmentation"]).reshape(-1, 2)
                points = [tuple(point) for point in points]
                # draw polygon, fill with label 1~32
                draw.polygon(points, fill=quadrant_remap[annotation["category_id"]] + 1)

        # save mask and copy image
        mask.save(f"dentex_dataset/segmentation/quadrant/masks/{image_info['file_name']}")
        shutil.copy(
            f"dentex_dataset/origin/quadrant/xrays/{image_info['file_name']}",
            f"dentex_dataset/segmentation/quadrant/xrays/{image_info['file_name']}",
        )

    save_json("dentex_dataset/segmentation/quadrant/image_names.json", image_names)


def process_seg_enumeration32():
    """
    draw segmentation masks for enumeration32
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")
    mkdirs("dentex_dataset/segmentation/enumeration32/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32/xrays")

    image_names = []
    for image_info in dataset_json["images"]:
        image_names.append(image_info["file_name"])
        # draw mask for each image
        image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for annotation in dataset_json["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                points = np.array(annotation["segmentation"]).reshape(-1, 2)
                points = [tuple(point) for point in points]
                # draw polygon, fill with label 1~32
                draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

        # save mask and copy image
        mask.save(f"dentex_dataset/segmentation/enumeration32/masks/{image_info['file_name']}")
        shutil.copy(
            f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
            f"dentex_dataset/segmentation/enumeration32/xrays/{image_info['file_name']}",
        )

    save_json("dentex_dataset/segmentation/enumeration32/image_names.json", image_names)


if __name__ == "__main__":
    process_seg_quadrant()
    process_seg_enumeration32()
    ...