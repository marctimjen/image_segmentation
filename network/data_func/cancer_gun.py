import cv2
import matplotlib.pyplot as plt

PATIENT_IX = "006"

img_f = f"/home/hp/Documents/GitHub/image_segmentation/data/all_data/patient_{PATIENT_IX}.png"
seg_f = f"/home/hp/Documents/GitHub/image_segmentation/data/all_data/segmentation_{PATIENT_IX}.png"

img = cv2.imread(img_f)
seg = cv2.imread(seg_f)


# Convert BGR images to RGB (Matplotlib uses RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

# Plot the images side by side
plt.figure(figsize=(12, 6))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')

# Plot the segmentation image
plt.subplot(1, 2, 2)
plt.imshow(seg)
plt.title('Segmentation Image')

plt.show()

