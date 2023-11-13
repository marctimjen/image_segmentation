from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import torch
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(992, 416)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'masks')

        self.images = os.listdir(self.image_folder)
        self.masks = os.listdir(self.mask_folder)

        assert len(self.images) == len(self.masks), "Number of images and masks should be the same."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.images[idx])
        if "patient" in self.images[idx]:
            mask_path = os.path.join(self.mask_folder, "segmentation_" + self.images[idx][-7:])
        else:
            mask_path = os.path.join(self.mask_folder, "target_seg_" + self.images[idx][-7:])

        # Load images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors
        tensor_image = torch.from_numpy(image)
        tensor_image = tensor_image.permute(2, 0, 1)

        tensor_mask = torch.from_numpy(mask)
        tensor_mask = tensor_mask.permute(2, 0, 1) / 255
        tensor_mask = tensor_mask[2:, :, :]

        # add padding
        pad_height = max(self.target_size[0] - tensor_image.size(1), 0)
        pad_width = max(self.target_size[1] - tensor_image.size(2), 0)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_image = transforms.functional.pad(tensor_image, (pad_left, pad_bottom, pad_right, pad_top), fill=255)
        padded_mask = transforms.functional.pad(tensor_mask, (pad_left, pad_bottom, pad_right, pad_top), fill=0)

        return torch.tensor(padded_image, dtype=torch.float32), torch.tensor(padded_mask, dtype=torch.float32)
