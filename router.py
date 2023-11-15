import numpy as np
from loguru import logger
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import torch.nn.functional as F

trans = transforms.Compose([transforms.ToTensor()])

# Model params
ENCODER_NAME = "resnet34"

# PATH = "network/models_save/CV-173/MODEL-Unetresnet34CV-173_EPOCH_76.pth"
# PATH = "network/models_save/CV-176/MODEL-Unetresnet34CV-176_EPOCH_83.pth"
# PATH = "network/models_save/CV-177/MODEL-Unetresnet34CV-177_EPOCH_48.pth"
PATH = "network/models_save/CV-179/MODEL-Unetresnet34CV-179_EPOCH_34.pth"

# Create the model
model = smp.Unet(
    encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                            # model output channels (number of classes in your dataset)
)

# model = smp.UnetPlusPlus(
#     encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,                            # model output channels (number of classes in your dataset)
# )

model.load_state_dict(torch.load(PATH))
model.eval()

router = APIRouter()

@router.post('/predict', response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):

    # Decode request str to numpy array
    img: np.ndarray = decode_request(request)

    # Obtain segmentation prediction
    predicted_segmentation = predict(img)

    # Validate segmentation format
    validate_segmentation(img, predicted_segmentation)

    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(predicted_segmentation)

    # Return the encoded segmentation to the validation/evalution service
    response = PredictResponseDto(
        img=encoded_segmentation
    )
    return response


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(img: np.ndarray) -> np.ndarray:
    # logger.info(f'Recieved image: {img.shape}')
    # threshold = 50
    # segmentation = get_threshold_segmentation(img, threshold)
    # return segmentation

    segmentation = unet_prediction(img)
    return segmentation

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


### DUMMY MODEL ###
def get_threshold_segmentation(img:np.ndarray, threshold:int) -> np.ndarray:
    return (img < threshold).astype(np.uint8)*255

