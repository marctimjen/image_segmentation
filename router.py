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
# ENCODER_NAME = "efficientnet-b3"
ENCODER_NAME = "resnet34"
# ENCODER_NAME = "resnet18"
# ENCODER_NAME = "resnext50_32x4d"


# PATH = "network/models_save/CV-173/MODEL-Unetresnet34CV-173_EPOCH_76.pth"
# PATH = "network/models_save/CV-176/MODEL-Unetresnet34CV-176_EPOCH_83.pth"
# PATH = "network/models_save/CV-177/MODEL-Unetresnet34CV-177_EPOCH_48.pth"
# PATH = "network/models_save/CV-179/MODEL-Unetresnet34CV-179_EPOCH_34.pth"
# PATH = "network/models_save/CV-199/MODEL-Unetefficientnet-b3CV-199_EPOCH_98.pth"

# PATH = "network/models_save/CV-202/MODEL-Unetresnet34CV-202_EPOCH_43.pth"
# PATH = "network/models_save/CV-204/MODEL-Unetresnet34CV-204_EPOCH_57.pth"

# PATH = "network/models_save/CV-209/MODEL-Unetresnet34CV-209_EPOCH_66.pth"
# PATH = "network/models_save/CV-211/MODEL-Unetresnet34CV-211_EPOCH_68.pth"


# PATH = "network/models_save/CV-212/MODEL-Unetresnet34CV-212_EPOCH_32.pth"  # 0.735,  0.7209
# PATH = "network/models_save/CV-212/MODEL-Unetresnet34CV-212_EPOCH_66.pth"  # 0.7348, 0.603
# PATH = "network/models_save/CV-213/MODEL-Unetresnet34CV-213_EPOCH_64.pth"
# PATH = "network/models_save/CV-214/MODEL-Unetresnet18CV-214_EPOCH_69.pth"
# PATH = "network/models_save/CV-215/MODEL-Unetresnet34CV-215_EPOCH_35.pth"


# PATH = "network/models_save/CV-216/MODEL-Unetresnet34CV-216_EPOCH_38.pth"
# PATH = "network/models_save/CV-216/MODEL-Unetresnet34CV-216_EPOCH_41.pth"
# PATH = "network/models_save/CV-216/MODEL-Unetresnet34CV-216_EPOCH_71.pth"


# PATH = "network/models_save/CV-220/MODEL-Unetresnet34CV-220_EPOCH_35.pth"
# PATH = "network/models_save/CV-220/MODEL-Unetresnet34CV-220_EPOCH_98.pth"

# PATH = "network/models_save/CV-222/MODEL-Unetresnext50_32x4dCV-222_EPOCH_27.pth"

# PATH = "network/models_save/CV-224/MODEL-Unetresnet34CV-224_EPOCH_46.pth"
# PATH = "network/models_save/CV-224/MODEL-Unetresnet34CV-224_EPOCH_32.pth"
# PATH = "network/models_save/CV-225/MODEL-Unetresnet34CV-225_EPOCH_68.pth"

# PATH = "network/models_save/CV-234/MODEL-Unetresnet34CV-234_EPOCH_67.pth"

# PATH = "network/models_save/CV-239/MODEL-Unetresnet34CV-239_EPOCH_21.pth"
# PATH = "network/models_save/CV-239/MODEL-Unetresnet34CV-239_EPOCH_50.pth"
# PATH = "network/models_save/CV-239/MODEL-Unetresnet34CV-239_EPOCH_79.pth"
# PATH = "network/models_save/CV-239/MODEL-Unetresnet34CV-239_EPOCH_81.pth"
# PATH = "network/models_save/CV-239/MODEL-Unetresnet34CV-239_EPOCH_11.pth"



# PATH = "network/models_save/CV-243/MODEL-Unetresnet34CV-243_EPOCH_17.pth"   # 0.546
# PATH = "network/models_save/CV-243/MODEL-Unetresnet34CV-243_EPOCH_30.pth"  # 0.664
# PATH = "network/models_save/CV-243/MODEL-Unetresnet34CV-243_EPOCH_71.pth"  #  0.673
# PATH = "network/models_save/CV-243/MODEL-Unetresnet34CV-243_EPOCH_83.pth"  #  0.7229
# PATH = "network/models_save/CV-243/MODEL-Unetresnet34CV-243_EPOCH_88.pth"  #  0.708

# PATH = "network/models_save/CV-244/MODEL-Unetresnet34CV-244_EPOCH_19.pth"  #  0.569
# PATH = "network/models_save/CV-244/MODEL-Unetresnet34CV-244_EPOCH_60.pth"  #  .7139

# PATH = "network/models_save/CV-248/MODEL-Unetresnet34CV-248_EPOCH_24.pth"  # 0.727

# PATH = "network/models_save/CV-248/MODEL-Unetresnet34CV-248_EPOCH_57.pth"  # 0.731,  83.5

# PATH = "network/models_save/CV-246/MODEL-Unetresnet34CV-246_EPOCH_47.pth"  #  0.702
# PATH = "network/models_save/CV-246/MODEL-Unetresnet34CV-246_EPOCH_54.pth"  #  0.702

# PATH = "network/models_save/CV-250/MODEL-Unetresnet34CV-250_EPOCH_55.pth"  # 0.589
# PATH = "network/models_save/CV-250/MODEL-Unetresnet34CV-250_EPOCH_99.pth"  # 0.591

# PATH = "network/models_save/CV-251/MODEL-Unetresnet34CV-251_EPOCH_30.pth"  # 0.709
# PATH = "network/models_save/CV-251/MODEL-Unetresnet34CV-251_EPOCH_51.pth"  # 0.674
# PATH = "network/models_save/CV-251/MODEL-Unetresnet34CV-251_EPOCH_98.pth"  # 0.687

# PATH = "network/models_save/CV-252/MODEL-Unetresnet34CV-252_EPOCH_34.pth"  # 0.718

# PATH = "network/models_save/CV-253/MODEL-Unetresnet34CV-253_EPOCH_17.pth"  # 0.723
# PATH = "network/models_save/CV-253/MODEL-Unetresnet34CV-253_EPOCH_44.pth"  # 0.693
# PATH = "network/models_save/CV-254/MODEL-Unetresnet34CV-254_EPOCH_28.pth"  # .721
# PATH = "network/models_save/CV-254/MODEL-Unetresnet34CV-254_EPOCH_61.pth"  # 0.724
# PATH = "network/models_save/CV-256/MODEL-Unetresnet34CV-256_EPOCH_97.pth"  #  0.707
# PATH = "network/models_save/CV-258/MODEL-Unetresnet34CV-258_EPOCH_20.pth"  #  0.670

# PATH = "network/models_save/CV-264/MODEL-Unetresnet34CV-264_EPOCH_26.pth"  # 0.676
# PATH = "network/models_save/CV-265/MODEL-Unetresnet34CV-265_EPOCH_23.pth"  # 0.6944

PATH = "network/models_save/CV-248/MODEL-Unetresnet34CV-248_EPOCH_31.pth"  # 0.734,  84  0.734  0.734

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

