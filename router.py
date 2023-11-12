import numpy as np
from loguru import logger
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
import segmentation_models_pytorch as smp
import torch


# Model params
ENCODER_NAME = "resnet34"

PATH = "CV-20/MODEL-Unetresnet34CV-20EPOCH8.pth"

# Create the model
model = smp.Unet(
    encoder_name=ENCODER_NAME,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                            # model output channels (number of classes in your dataset)
)
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
    logger.info(f'Recieved image: {img.shape}')
    threshold = 50
    segmentation = get_threshold_segmentation(img, threshold)
    return segmentation

def unet_prediction() -> np.ndarray:
    pass

### DUMMY MODEL ###
def get_threshold_segmentation(img:np.ndarray, threshold:int) -> np.ndarray:
    return (img < threshold).astype(np.uint8)*255

