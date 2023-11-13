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
import optuna

from network_funcs.dataset import SegmentationDataset
from network_funcs.net_loss import DiceBCELoss
import neptune.integrations.optuna as optuna_utils

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
neptune_callback = optuna_utils.NeptuneCallback(run)
run_id = run["sys/id"].fetch()


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):

    opt_id = trial.number

    # Create the model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                            # model output channels (number of classes in your dataset)
        )

    # Training params
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 2, 14)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=LEARNING_RATE)

    EPOCHS = 100

    # Proper directories
    TRAIN_DATA_DIR = 'image_data/train'
    VAL_DATA_DIR = 'image_data/val'

    run_name = "MODEL-" + model.__class__.__name__ + ENCODER_NAME + str(run_id)

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

    run[f"trials/trials/{opt_id}/params"] = params

    model = model.to(device)

    # Set up dataset and dataloader
    transform = transforms.Compose([])

    trainDataset = SegmentationDataset(root_dir=TRAIN_DATA_DIR, transform=transform)
    valDataset = SegmentationDataset(root_dir=VAL_DATA_DIR, transform=transform)

    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss = torch.tensor(9999999.0)
    # best_dice = torch.tensor(0.0)
    count = 0

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
        run[f"trials/trials/{opt_id}/loss/train_loss"].log(train_loss)

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
                dice_score += calc_dice_score(torch.sigmoid(model_outputs), val_masks_int, ignore_index=0).cpu()

                iou += calc_iou(model_outputs, val_masks_int).cpu()


        val_loss /= len(val_loader)
        iou /= len(val_loader)
        dice_score /= len(val_loader)

        if torch.isnan(iou):
            iou = torch.tensor(0.0)

        run[f"trials/trials/{opt_id}/loss/val_loss"].log(val_loss)
        run[f"trials/trials/{opt_id}/val/iou"].log(iou)
        run[f"trials/trials/{opt_id}/val/dice_score"].log(dice_score)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_dice = dice_score
            count = 0
        else:
            count += 1
            if count > 5:
                break


    return best_val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, callbacks=[neptune_callback])
# study.optimize(objective, n_trials=2, callbacks=[neptune_callback], timeout=25*60)

run.stop()
