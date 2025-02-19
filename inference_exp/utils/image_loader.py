import json
import os
from colorama import Fore, Style
import cv2

class Image_Loader:
    images = {}
    image_files = {}
    def __init__(self):
        None

    def load_image_filepaths(self, image_dir, start_id=1, end_id=-1):
        img_files = os.listdir(image_dir)
        img_files.sort()
        if end_id == -1:
            end_id = len(img_files)
        else:
            end_id = min(end_id, len(img_files))
        img_files = img_files[:end_id]
        for index, img_file in zip(range(len(img_files)), img_files):
            self.image_files[start_id + index] = image_dir + os.sep + img_file


    def load_images(self, image_dir, start_id=1, end_id=-1):
        self.load_image_filepaths(image_dir, start_id, end_id)
        for img_id, img_file in self.image_files.items():
            # check if the image file exists
            if not os.path.exists(img_file):
                print(Fore.RED + "Image file not found: ", img_file + Style.RESET_ALL)
                continue
            self.images[img_id] = cv2.imread(img_file)


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


    def get_image_filepath(self, image_id):
        if image_id not in self.image_files:
            print(Fore.RED + "Image not found" + Style.RESET_ALL)
            return None
        return self.image_files[image_id]


    def print_loaded_images(self):
        if len(self.images) == 0 or len(self.image_files) == 0:
            print(Fore.RED + "No images loaded" + Style.RESET_ALL)
            return

        if len(self.images) > 0:
            print(Fore.GREEN + "Images loaded and cached" + Style.RESET_ALL)
            for index, img in self.images.items():
                print(f"Image {index}: {img.shape} from {self.image_files[index]}")
        else:
            print(Fore.BLUE + "Only image filepaths are loaded" + Style.RESET_ALL)
            for index, img in self.image_files.items():
                print(f"Image {index}: {img}")


    def print_loaded_images_brief(self):
        if len(self.images) == 0 or len(self.image_files) == 0:
            print(Fore.RED + "No images loaded" + Style.RESET_ALL)
            return

        if len(self.images) > 0:
            # first item of self.image_files
            index, img_file = next(iter(self.image_files.items()))
            last_index, last_img_file = next(iter(reversed(self.image_files.items())))

            print(Fore.GREEN + f"# Images loaded and cached #" + Style.RESET_ALL)
            print(Fore.YELLOW + f"\tTotal {len(self.images)} images" + Style.RESET_ALL)
            print(Fore.YELLOW + f"\t\tfirst file index ({index}): {self.image_files[index]}" + Style.RESET_ALL)
            print(Fore.YELLOW + f"\t\tlast file index ({last_index}): {self.image_files[last_index]}" + Style.RESET_ALL)


        else:
            index, img_file = next(iter(self.image_files.items()))
            print(Fore.BLUE + "Only image filepaths are loaded" + Style.RESET_ALL)
            print(Fore.YELLOW + f"\tTotal {len(self.images)} images, first file index ({index}): {self.image_files[index]}" + Style.RESET_ALL)


if __name__ == "__main__":
    image_loader = Image_Loader()
    # image_loader.load_image_filepaths("/home/jin/mnt/github/mmdetection/inference_exp/sAP/data/img/")
    image_loader.load_images("/home/jin/mnt/github/mmdetection/inference_exp/data/img/")
    image_loader.print_loaded_images()
    image_loader.print_loaded_images_brief()
