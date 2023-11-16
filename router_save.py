import os

import numpy as np
from loguru import logger
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
import pickle
import numpy as np


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

    save_path = rf"/home/paperspace/save_pics"
    j = len(os.listdir(save_path))

    file_path = save_path + f"img_{j}.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(img, file)

    j += 1

    threshold = 50
    segmentation = get_threshold_segmentation(img, threshold)
    return segmentation

    # segmentation = unet_prediction(img)
    # return segmentation

# def unet_prediction(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
#     inp = torch.tensor(255 * trans(img), dtype=torch.float32)
#
#     pad_height = max(992 - inp.shape[1], 0)
#     pad_width = max(416 - inp.shape[2], 0)
#     pad_top = pad_height // 2
#     pad_bottom = pad_height - pad_top
#     pad_left = pad_width // 2
#     pad_right = pad_width - pad_left
#     padded_image = transforms.functional.pad(inp, (pad_left, pad_bottom, pad_right, pad_top), fill=255)
#
#     out = model(padded_image.unsqueeze(0))[0]  # get the feature map
#
#     # TODO: Test this unpadding
#
#     original_height = padded_image.shape[-2] - pad_top - pad_bottom
#     original_width = padded_image.shape[-1] - pad_left - pad_right
#
#     # Use slicing to unpad the image
#     out = out[:, pad_top:pad_top + original_height, pad_left:pad_left + original_width]
#     out = F.sigmoid(out)
#     out = (out >= threshold).numpy().astype(np.uint8)*255
#
#     segment_map = np.tile(out, (3, 1, 1))
#     segment_map = np.transpose(segment_map, (1, 2, 0))
#     return segment_map


### DUMMY MODEL ###
def get_threshold_segmentation(img:np.ndarray, threshold:int) -> np.ndarray:
    return (img < threshold).astype(np.uint8)*255

