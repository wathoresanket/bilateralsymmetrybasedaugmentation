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
import models.unet.utils as utils

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

    set_seeds(args.seed)
    cuda = args.cuda
    is_parallel = args.is_parallel

    num_classes = args.num_classes

    # model change
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = num_classes + 1
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    if cuda:
        model = model.cuda()
    if is_parallel:
        model = nn.DataParallel(model)

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.info("Loading dataset...")

    train_dataset_dir = args.train_dataset_dir
    val_dataset_dir = args.val_dataset_dir
    train_dataset = Preload(SegmentationDataset(train_dataset_dir))
    val_dataset = Preload(SegmentationDataset(val_dataset_dir))

    logger.info("Loaded dataset!")

    # # Calculate mean and std
    # mean_train, std_train = calculate_mean_std(train_dataset_dir, os.path.join(train_dataset_dir, "image_names.json"))
    # mean_val, std_val = calculate_mean_std(val_dataset_dir, os.path.join(val_dataset_dir, "image_names.json"))

    # Use predefined mean and std
    mean_train, std_train = [0.458], [0.173]
    mean_val, std_val = [0.458], [0.173]

    train_dataset = TransformedDataset(train_dataset, flip=0.1, crop=0.1, rotate=[-10, 10], mean=mean_train, std=std_train)
    val_dataset = TransformedDataset(val_dataset, flip=0.1, crop=0.1, rotate=[-10, 10], mean=mean_val, std=std_val)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # model change
    dice_loss_func = DiceLoss(config_vit.n_classes)

    ce_loss_func = CrossEntropyLoss()

    tensorboard_writer = SummaryWriter(output_dir)

    if args.resume is not None:
        logger.warning("Resume not implemented yet, ignoring it!")

    min_valid_loss = float("inf")
    best_model_path = None
    for epoch in range(args.epochs):
        logger.info(f"Epoch: {epoch}")

        model.train()
        train_loss_dice = 0.0
        train_loss_ce = 0.0
        for i, (image, mask) in enumerate(train_loader):
            logger.info(f"Train batch: {i}/{len(train_loader)}")
            if cuda:
                image = image.cuda()
                mask = mask.cuda()

            optimizer.zero_grad()
            pred = model(image)
            loss_dice = dice_loss_func(pred, mask, softmax=True)
            loss_ce = ce_loss_func(pred, mask.long())
            loss = loss_dice + loss_ce
            loss.backward()
            optimizer.step()

            train_loss_dice += loss_dice.item()
            train_loss_ce += loss_ce.item()

        train_loss_dice /= len(train_loader)
        train_loss_ce /= len(train_loader)
        logger.info(f"Train loss: {train_loss_dice}, {train_loss_ce}")
        tensorboard_writer.add_scalar("train_loss/total", train_loss_dice + train_loss_ce, epoch)
        tensorboard_writer.add_scalar("train_loss/dice", train_loss_dice, epoch)
        tensorboard_writer.add_scalar("train_loss/ce", train_loss_ce, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss_dice = 0.0
            valid_loss_ce = 0.0
            for i, (image, mask) in enumerate(val_loader):
                logger.info(f"Validation batch: {i}/{len(val_loader)}")
                if cuda:
                    image = image.cuda()
                    mask = mask.cuda()

                pred = model(image)
                loss_dice = dice_loss_func(pred, mask, softmax=True)
                loss_ce = ce_loss_func(pred, mask.long())

                valid_loss_dice += loss_dice.item()
                valid_loss_ce += loss_ce.item()

            valid_loss_dice /= len(val_loader)
            valid_loss_ce /= len(val_loader)
            logger.info(f"Validation loss: {valid_loss_dice}, {valid_loss_ce}")
            tensorboard_writer.add_scalar("validation_loss/total", valid_loss_dice + valid_loss_ce, epoch)
            tensorboard_writer.add_scalar("validation_loss/dice", valid_loss_dice, epoch)
            tensorboard_writer.add_scalar("validation_loss/ce", valid_loss_ce, epoch)

            if valid_loss_dice + valid_loss_ce < min_valid_loss:
                min_valid_loss = valid_loss_dice + valid_loss_ce
                
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = os.path.join(output_dir, f"epoch_{epoch}_loss_{min_valid_loss}.pth")

                utils.save_state(
                    model=model,
                    out_dir=output_dir,
                    checkpoint_name=f"epoch_{epoch}_loss_{min_valid_loss}.pth",
                    batch_size=batch_size,
                    epoch=epoch,
                    is_parallel=is_parallel,
                )
                logger.info(f"Save model: epoch_{epoch}_loss_{min_valid_loss}.pth")

    tensorboard_writer.close()
    logger.info("Done!")

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_dir", type=str)
    parser.add_argument("--val_dataset_dir", type=str)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--is_parallel", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--model", type=str, default='ViT_seg', help="Model name")
    parser.add_argument("--num_classes", type=int, help="number of classes, not including background")
    # model change
    parser.add_argument("--img_size", type=int, default=256, help="input patch size of network input")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
