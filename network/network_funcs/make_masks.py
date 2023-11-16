import numpy as np
import shutil
import cv2
import os


def process_images(source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of image files in the source folder
    image_files = os.listdir(source_folder)
    i = 500
    for img_file in image_files:
        # Read the image
        img_path = os.path.join(source_folder, img_file)
        image = cv2.imread(img_path)

        if image is not None:
            # Get image dimensions
            height, width, _ = image.shape

            # Create an empty image with the same dimensions
            empty_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Save the empty image
            empty_img_path = os.path.join(destination_folder, f"target_seg_{str(i)}.png")
            cv2.imwrite(empty_img_path, empty_image)

            # Define the new name for the image (you can adjust the naming convention)
            new_name = f"control_{str(i)}.png"

            # Define the path for the new image in the destination folder
            new_img_path = os.path.join(destination_folder, new_name)

            # Move and rename the original image to the destination folder
            shutil.move(img_path, new_img_path)

            i += 1



# pic = cv2.imread("/home/hp/Documents/data/pics_after/target_seg_500.png")
# print()

if __name__ == "__main__":
    source_folder = "/home/hp/Documents/data/control_images_inv"
    destination_folder = "/home/hp/Documents/data/pics_after"

    process_images(source_folder, destination_folder)
