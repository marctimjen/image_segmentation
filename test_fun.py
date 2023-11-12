#%%
from matplotlib import pyplot as plt
from utils import dice_score, plot_prediction
from torchmetrics.functional.classification import dice as calc_dice_score
from router import predict
import cv2
import torch
from torchvision import transforms
from utils import validate_segmentation

trans = transforms.Compose([transforms.ToTensor()])

PATIENT_IX = "006"

img_f = f"/home/hp/Documents/GitHub/image_segmentation/data/all_data/patient_{PATIENT_IX}.png"
seg_f = f"/home/hp/Documents/GitHub/image_segmentation/data/all_data/segmentation_{PATIENT_IX}.png"

img = cv2.imread(img_f)

out = predict(img)
validate_segmentation(img, out)



print()



# seg = cv2.imread(seg_f)
# seg_pred = predict(img)


# target = 1 * ((torch.randn((4, 3, 300, 400)) - 0.5) > 0)
#
# pred = torch.randn((4, 3, 300, 400))
#
# # calc_dice_score = dice()
# print("dice_score dice", dice_score(1 * (pred> 0), target))
# m = torch.nn.Sigmoid()
# print("Torch dice", calc_dice_score(m(pred), target, ignore_index=0))
#
#
#
#
# print("dice_score dice", dice_score(torch.tensor(trans(seg_pred)[2, :, :], dtype=torch.int8), torch.tensor(trans(seg)[2, :, :], dtype=torch.int8)))
#
# print("Torch dice", calc_dice_score(torch.tensor(trans(seg_pred), dtype=torch.int8), torch.tensor(trans(seg), dtype=torch.int8), ignore_index=0))
#
# plot_prediction(img,seg,seg_pred)

