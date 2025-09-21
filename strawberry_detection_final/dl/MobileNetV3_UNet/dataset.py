import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

class StrawSegDataset(Dataset):
    """
    Dataset for strawberry ROI segmentation.
    Expects image and mask directories with matching filenames.
    """
    def __init__(self, images_dir, masks_dir, img_size=128, augment=False):
        self.img_paths = sorted(
            glob.glob(os.path.join(images_dir, '*.png')) +
            glob.glob(os.path.join(images_dir, '*.jpg'))
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(masks_dir, '*.png'))
        )
        assert len(self.img_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) must match"
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Resize
        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))

        # Augmentation: horizontal flip
        if self.augment and torch.rand(1) < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # To tensor
        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask
