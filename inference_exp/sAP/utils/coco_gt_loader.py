import json
from colorama import Fore, Style

class COCO_GT_Loader:
    images = {}
    annotations = {}
    categories = {}
    def __init__(self):
        None

    def load_gts(self, gt_json_file):
        with open(gt_json_file) as f:
            gts_json = json.load(f)
            print("Loading GTs from file: ", gt_json_file)

            # load image info
            for image in gts_json['images']:
                self.images[image['id']] = image['file_name']

            # laod annotations
            for annotation in gts_json['annotations']:
                image_index = annotation['image_id']
                category_index = annotation['category_id']
                bbox = annotation['bbox']

                if image_index not in self.annotations:
                    self.annotations[image_index] = []

                self.annotations[image_index].append({
                    'category': category_index,
                    'bbox': bbox
                })

            # load categories
            for category in gts_json['categories']:
                self.categories[category['id']] = category['name']

    def print_image_info(self):
        print(Fore.GREEN + 'PRINT IMAGE INFO' + Style.RESET_ALL)
        for image_id, image_name in self.images.items():
            print("Image ID: ", image_id, " Image Name: ", image_name)

    def print_annotation_info(self):
        print(Fore.GREEN + 'PRINT ANNOTATION INFO' + Style.RESET_ALL)
        for image_id, annotations in self.annotations.items():
            print("Image ID: ", image_id, " Annotations: ", annotations)

    def print_category_info(self):
        print(Fore.GREEN + 'PRINT CATEGORY INFO' + Style.RESET_ALL)
        for category_id, category_name in self.categories.items():
            print("Category ID: ", category_id, " Category Name: ", category_name)

if __name__ == "__main__":
    coco_gt_loader = COCO_GT_Loader()
    coco_gt_loader.load_gts("/home/jin/mnt/github/mmdetection/inference_exp/sAP/data/gt/annotation.json")
    coco_gt_loader.print_image_info()
    coco_gt_loader.print_annotation_info()

