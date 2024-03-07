import json
from colorama import Fore, Style

class COCO_GT_Loader:
    images = {}
    annotations = {}
    categories = {}
    def __init__(self):
        None


    def load_gts(self, gt_json_file, end_id=-1):
        with open(gt_json_file) as f:
            gts_json = json.load(f)
            print("Loading GTs from file: ", gt_json_file)

            # load image info
            for image in gts_json['images']:
                self.images[image['id']] = image['file_name']

            # laod annotations
            for annotation in gts_json['annotations']:
                image_id = annotation['image_id']
                category_index = annotation['category_id']
                bbox = annotation['bbox']

                if image_id not in self.annotations:
                    self.annotations[image_id] = []

                self.annotations[image_id].append({
                    'category': category_index,
                    'bbox': bbox
                })

            # filter out image and annotation over end_id
            if end_id > 0:
                filtered_images = {}
                filtered_annotations = {}
                for image_id, image_name in self.images.items():
                    if image_id > end_id:
                        continue
                    filtered_images[image_id] = image_name
                    filtered_annotations[image_id] = self.annotations[image_id]
                self.images = filtered_images
                self.annotations = filtered_annotations

            # load categories
            for category in gts_json['categories']:
                self.categories[category['id']] = category['name']


    def get_annotation_for_evaluation(self, image_id):
        if image_id not in self.annotations:
            return None

        eval_annot = []
        for annotation in self.annotations[image_id]:
            # print(annotation)
            eval_annot.append({'image_id': image_id, 'bbox_label': annotation['category'], 'bbox': annotation['bbox']})

        return eval_annot


    def get_category_names(self):
        category_names = []
        for category_id, category_name in self.categories.items():
            category_names.append(category_name)
        return category_names


    def get_first_image_id(self):
        img_ids = list(self.images.keys())
        img_ids.sort()
        return img_ids[0]


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


    def print_loaded_gt_brief(self):
        if len(self.images) == 0 or len(self.annotations) == 0 or len(self.categories) == 0:
            print(Fore.RED + "No GTs loaded" + Style.RESET_ALL)
            return

        print(Fore.GREEN + "# GTs loaded #" + Style.RESET_ALL)
        image_id, image_name = next(iter(self.images.items()))
        last_image_id, last_image_name = next(iter(reversed(self.images.items())))
        print(Fore.YELLOW + f"\tTotal {len(self.images)} images" + Style.RESET_ALL)
        print(Fore.YELLOW + f"\t\tfirst image id ({image_id}): {image_name}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"\t\tlast image id ({last_image_id}): {last_image_name}" + Style.RESET_ALL)

        annotation_count = 0
        for image_id, annotations in self.annotations.items():
            annotation_count += len(annotations)
        print(Fore.YELLOW + f"\tTotal {annotation_count} annotations" + Style.RESET_ALL)

        category_count = len(self.categories)
        print(Fore.YELLOW + f"\tTotal {category_count} categories" + Style.RESET_ALL)




if __name__ == "__main__":
    coco_gt_loader = COCO_GT_Loader()
    coco_gt_loader.load_gts("/home/jin/mnt/github/mmdetection/inference_exp/data/gt/annotation.json")
    coco_gt_loader.print_image_info()
    coco_gt_loader.print_annotation_info()
    annot1 = coco_gt_loader.get_annotation_for_evaluation(0)
    annot2 = coco_gt_loader.get_annotation_for_evaluation(1)
    print(annot1)
    print(annot2)
    coco_gt_loader.print_category_info()
    first_img_id = coco_gt_loader.get_first_image_id()
    print("First image id: ", first_img_id)

    coco_gt_loader.print_loaded_gt_brief()
