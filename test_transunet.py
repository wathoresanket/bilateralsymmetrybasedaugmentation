import os
import json
import time
import logging
import random
import argparse
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms, InterpolationMode
import torchvision.transforms.functional as ttf

from models.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.transunet.utils import DiceLoss


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, "image_names.json"), "r") as f:
            self.image_names = json.load(f)

    def __getitem__(self, index) -> tuple[Image.Image, Image.Image]:
        image_name = self.image_names[index]
        with Image.open(os.path.join(self.dataset_dir, "xrays", image_name)) as image:
            image = image.convert("L").copy()
        with Image.open(os.path.join(self.dataset_dir, "masks", image_name)) as mask:
            mask = mask.copy()

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)

class Preload(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = []
        for i in range(len(dataset)):
            self.data.append(dataset[i])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.dataset)

class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        flip: float = None,
        crop: float = None,
        rotate: list = None,
        mean: list = None,
        std: list = None,
    ):
        self.dataset = dataset
        self.flip = flip
        self.crop = crop
        self.rotate = rotate
        self.mean = mean
        self.std = std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        image, mask = self.data_transform(image, mask)
        return image, mask

    def __len__(self) -> int:
        return len(self.dataset)

    def data_transform(
        self, image: Image.Image, mask: Image.Image = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert PIL Image to torch Tensor and do some augmentation
        @param image: PIL Image
        @param mask: PIL Image
        @param flip: float, 0.0 ~ 1.0, probability of flip
        @param crop: float, 0.0 ~ 1.0, probability of crop
        @param rotate: list, [min_angle, max_angle], in degree
        """
        dummy_mask = mask if mask is not None else Image.new("L", image.size)
        # resize
        image = image.resize((256, 256), Image.BILINEAR)
        dummy_mask = dummy_mask.resize((256, 256), Image.NEAREST)

        # to tensor
        image = ttf.to_tensor(image)  # shape(1, 256, 256)
        dummy_mask = torch.from_numpy(np.array(dummy_mask)).long().unsqueeze(0)  # shape(1, 256, 256)

        # normalize
        image = ttf.normalize(image, self.mean, self.std)

        # # flip
        # if self.flip is not None and random.random() < self.flip:
        #     image = ttf.hflip(image)
        #     dummy_mask = ttf.hflip(dummy_mask)

        # # crop
        # if self.crop is not None and random.random() < self.crop:
        #     size = random.randint(128, 225)
        #     i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(size, size))
        #     image = ttf.crop(image, i, j, h, w)
        #     dummy_mask = ttf.crop(dummy_mask, i, j, h, w)

        #     # resize
        #     image = ttf.resize(image, (256, 256), InterpolationMode.BILINEAR)
        #     dummy_mask = ttf.resize(dummy_mask, (256, 256), InterpolationMode.NEAREST)

        # # rotate
        # if self.rotate is not None and random.random() < 0.1:
        #     angle = random.randint(self.rotate[0], self.rotate[1])
        #     image = ttf.rotate(image, angle)
        #     dummy_mask = ttf.rotate(dummy_mask, angle)

        dummy_mask = dummy_mask.squeeze(0)
        return image, dummy_mask

def calculate_mean_std(dataset_dir: str, json_file: str):
    with open(json_file, "r") as f:
        image_names = json.load(f)
    
    mean = 0.0
    std = 0.0
    num_images = len(image_names)

    for image_name in image_names:
        image = Image.open(os.path.join(dataset_dir, "xrays", image_name)).convert("L")
        image = np.array(image) / 255.0
        mean += image.mean()
        std += image.std()

    mean /= num_images
    std /= num_images
    return [mean], [std]

def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(__file__, output_dir)

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(output_dir, "testing.log"))
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    set_seeds(args.seed)
    cuda = args.cuda
    is_parallel = args.is_parallel

    num_classes = args.num_classes

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = num_classes + 1
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Load the saved model
    model_path = args.model_path
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}.")
        return
    
    if cuda:
        model = model.cuda()
    if is_parallel:
        model = nn.DataParallel(model)

    logger.info("Loading dataset...")

    test_dataset_dir = args.dataset_dir
    test_dataset = Preload(SegmentationDataset(test_dataset_dir))

    logger.info("Loaded dataset!")

    # # Calculate mean and std
    # mean, std = calculate_mean_std(test_dataset_dir, os.path.join(test_dataset_dir, "image_names.json"))

    # Use predefined mean and std
    mean, std = [0.458], [0.173]

    test_dataset = TransformedDataset(test_dataset, mean=mean, std=std)

    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dice_loss_func = DiceLoss(config_vit.n_classes)
    ce_loss_func = CrossEntropyLoss()

    dsc_values = []  # Initialize an empty list to store DSC values
    
    logger.info(f"Start testing...")

    model.eval()
    with torch.no_grad():
        test_loss_dice = 0.0
        test_loss_ce = 0.0
        for i, (image, mask) in enumerate(test_loader):
            logger.info(f"Testing batch: {i}/{len(test_loader)}")
            if cuda:
                image = image.cuda()
                mask = mask.cuda()

            pred = model(image)

            loss_dice = dice_loss_func(pred, mask, softmax=True)
            loss_ce = ce_loss_func(pred, mask)

            test_loss_dice += loss_dice.item()
            test_loss_ce += loss_ce.item()

            # Calculate DSC
            dsc = 1 - loss_dice.item()  # DSC = 1 - Dice Loss
            dsc_values.append(dsc)  # Append the DSC value to the list

        test_loss_dice /= len(test_loader)
        test_loss_ce /= len(test_loader)
        logger.info(f"Testing loss: {test_loss_dice}, {test_loss_ce}")
        logger.info(f"Testing Dice Loss: {test_loss_dice}")
        logger.info(f"Testing Cross Entropy Loss: {test_loss_ce}")

    logger.info(f"DSC values for all images: {dsc_values}")  # Log the DSC values

    logger.info("Done!")

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--is_parallel", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--model", type=str, default='ViT_seg', help="Model name")
    parser.add_argument("--num_classes", type=int, help="number of classes, not including background")
    # model change
    parser.add_argument("--model_path", type=str, help="Path to the saved model (.pth file)")
    parser.add_argument("--img_size", type=int, default=256, help="input patch size of network input")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
