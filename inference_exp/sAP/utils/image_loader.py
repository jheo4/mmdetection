import json
import os
from colorama import Fore, Style
import cv2

class Image_Loader:
    images = {}
    image_files = {}
    def __init__(self):
        None

    def load_image_filepaths(self, image_dir):
        img_files = os.listdir(image_dir)
        img_files.sort()
        for index, img_file in zip(range(len(img_files)), img_files):
            self.image_files[index] = image_dir + os.sep + img_file


    def load_images(self, image_dir):
        self.load_image_filepaths(image_dir)
        for index, img_file in self.image_files.items():
            self.images[index] = cv2.imread(img_file)


    def get_image(self, index):
        if index not in self.images or index not in self.image_files:
            print(Fore.RED + "Image not found" + Style.RESET_ALL)
            return None

        if index in self.images:
            return self.images[index]

        if index in self.image_files:
            self.images[index] = cv2.imread(self.image_files[index])
            return self.images[index]

        return None


    def print_loaded_images(self):
        if len(self.images) == 0 or len(self.image_files) == 0:
            print(Fore.RED + "No images loaded" + Style.RESET_ALL)
            return

        if len(self.images) > 0:
            print(Fore.GREEN + "Images loaded and cached" + Style.RESET_ALL)
            for index, img in self.images.items():
                print(f"Image {index}: {img.shape}")
        else:
            print(Fore.BLUE + "Only image filepaths are loaded" + Style.RESET_ALL)
            for index, img in self.image_files.items():
                print(f"Image {index}: {img}")


if __name__ == "__main__":
    image_loader = Image_Loader()
    # image_loader.load_image_filepaths("/home/jin/mnt/github/mmdetection/inference_exp/sAP/data/img/")
    image_loader.load_images("/home/jin/mnt/github/mmdetection/inference_exp/sAP/data/img/")

