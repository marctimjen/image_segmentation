import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torch.nn as nn
import time
import copy
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
import neptune
from neptune.types import File

from torchmetrics.functional.classification import dice as calc_dice_score
from torchmetrics.classification import BinaryJaccardIndex

# Define a custom dataset class
# Training params
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

# Model params
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

run = neptune.init_run(
    project="Kernel-bois/computer-vision",
    api_token=token,
)
run_id = run["sys/id"].fetch()

# Create the model
model = smp.Unet(
    encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                            # model output channels (number of classes in your dataset)
    )


# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#Scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

run_name = "MODEL-" + model.__class__.__name__ + ENCODER_NAME + str(run_id)

save_path = str(run_id) + "/"
os.makedirs(save_path)

# Proper directories
TRAIN_DATA_DIR = 'image_data/train'
VAL_DATA_DIR = 'image_data/val'

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE



# Define loss function
# criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)  # Binary dice Loss for binary segmentation
# criterion = smp.losses.SoftBCEWithLogitsLoss()  # Binary dice Loss for binary segmentation
criterion = DiceBCELoss()  # Binary dice Loss for binary segmentation
calc_iou = BinaryJaccardIndex().to(device)


params = {
    "MODEL": model.__class__.__name__,
    "BACKBONE": ENCODER_NAME,
    "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
    "BATCH_SIZE": str(BATCH_SIZE),
    "EPOCHS": str(EPOCHS),
    "CRITERION": criterion.__class__.__name__,
    "OPTIMIZER": optimizer.__class__.__name__,
    "LEARNRATE": str(LEARNING_RATE),
    "MODEL_NAME": run_name,
}

run["params"] = params

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform = None, target_size = (992, 416)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'masks')

        self.images = os.listdir(self.image_folder)
        self.masks = os.listdir(self.mask_folder)

        assert len(self.images) == len(self.masks), "Number of images and masks should be the same."

        self.imageConverter = transforms.Compose([transforms.PILToTensor()])
        self.maskConverter = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.images[idx])
        if "patient" in self.images[idx]:
            mask_path = os.path.join(self.mask_folder, "segmentation_" + self.images[idx][-7:])
        else:
            mask_path = os.path.join(self.mask_folder, "target_seg_" + self.images[idx][-7:])

        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Convert to tensors
        tensor_image = self.imageConverter(image)
        tensor_mask = self.maskConverter(mask)
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

        if self.transform:
            padded_image = self.transform(padded_image)
            padded_mask = self.transform(padded_mask)

        return torch.tensor(padded_image, dtype=torch.float32), torch.tensor(padded_mask, dtype=torch.float32)

model = model.to(device)

# Set up dataset and dataloader
transform = transforms.Compose([])

trainDataset = SegmentationDataset(root_dir=TRAIN_DATA_DIR, transform=transform)
valDataset = SegmentationDataset(root_dir=VAL_DATA_DIR, transform=transform)

train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Train Loop
for epoch in range(EPOCHS):
    train_loss = torch.tensor(0.0)
    model.train()

    # Use tqdm to add a progress bar
    for images, masks in train_loader: #tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        # outputs = torch.argmax(outputs, dim=1).unsqueeze(1).float()

        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.detach().cpu()
        optimizer.step()

    train_loss /= len(train_loader)
    run["loss/train_loss"].log(train_loss)

    # Validation
    model.eval()

    val_loss = torch.tensor(0.0)
    iou = torch.tensor(0.0)
    dice_score = torch.tensor(0.0)

    with torch.no_grad():
        for val_images, val_masks in val_loader:  # tqdm(val_loader, desc=f'Validation', leave=False):
            val_images, val_masks = val_images.to(device), val_masks.to(device)

            model_outputs = model(val_images)

            val_loss += criterion(model_outputs, val_masks).cpu()

            val_masks_int = torch.tensor(val_masks, dtype=torch.int8)
            dice_score += calc_dice_score(F.sigmoid(model_outputs), val_masks_int, ignore_index=0).cpu()

            iou += calc_iou(model_outputs, val_masks_int).cpu()


    val_loss /= len(val_loader)
    iou /= len(val_loader)
    dice_score /= len(val_loader)

    if torch.isnan(iou):
        iou = torch.tensor(0.0)

    run["loss/val_loss"].log(val_loss)
    run["val/iou"].log(iou)
    run["val/dice_score"].log(dice_score)

    torch.save(model.state_dict(), save_path + run_name + "_EPOCH_" + str(epoch) + '.pth')

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss}, Validation Loss: {val_loss}\n"
          f"IOU: {iou}, Dice Score: {dice_score}")

# Save the trained model
torch.save(model.state_dict(), save_path + run_name + "_FINAL" + '.pth')
run[f"network/network_weights"].upload(File(run_name + '.pth'))

run.stop()
