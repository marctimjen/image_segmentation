import os

import numpy as np

from models.dtos import PredictRequestDto, PredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import torch.nn.functional as F
trans = transforms.Compose([transforms.ToTensor()])
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl


plt.ion()
mpl.use("TkAgg")



path_to_pics = "/home/hp/Documents/Val_pics/save_pics"

val_pics = os.listdir(path_to_pics)


# Model params
# ENCODER_NAME = "efficientnet-b3"
ENCODER_NAME = "resnet34"
# ENCODER_NAME = "resnet18"
# ENCODER_NAME = "resnext50_32x4d"



model = smp.Unet(
    encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                            # model output channels (number of classes in your dataset)
)

model_path = rf"/home/hp/Documents/GitHub/image_segmentation/network/models_save/CV-212/MODEL-Unetresnet34CV-212_EPOCH_66.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

def unet_prediction(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    inp = torch.tensor(255 * trans(img), dtype=torch.float32)

    pad_height = max(992 - inp.shape[1], 0)
    pad_width = max(416 - inp.shape[2], 0)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_image = transforms.functional.pad(inp, (pad_left, pad_bottom, pad_right, pad_top), fill=255)

    out = model(padded_image.unsqueeze(0))[0]  # get the feature map

    # TODO: Test this unpadding

    original_height = padded_image.shape[-2] - pad_top - pad_bottom
    original_width = padded_image.shape[-1] - pad_left - pad_right

    # Use slicing to unpad the image
    out = out[:, pad_top:pad_top + original_height, pad_left:pad_left + original_width]
    out = F.sigmoid(out)
    out = (out >= threshold).numpy().astype(np.uint8)*255

    segment_map = np.tile(out, (3, 1, 1))
    segment_map = np.transpose(segment_map, (1, 2, 0))
    return segment_map



def plot_image_and_mask(image, mask):
    image = image / 255.0
    mask = mask

    # Plot side by side with the mask and mask overlain
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')

    # Plot the mask
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Mask')

    # Overlay the mask on the image
    axes[2].imshow(image)
    axes[2].imshow(mask, cmap='viridis', alpha=0.6)  # Set alpha to less than 1
    axes[2].set_title('Mask Overlain on Image')

    # Display the plots
    plt.show()




for pic in val_pics:
    with open(path_to_pics + "/" + pic, 'rb') as file:
        loaded_array = pkl.load(file)

    unet_pred = unet_prediction(loaded_array)

    plot_image_and_mask(loaded_array, unet_pred)
    print()




# predict_image(model, image, mask)
