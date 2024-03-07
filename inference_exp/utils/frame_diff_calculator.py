import cv2
import piq
import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image

class Frame_Diff_Calculator:
    most_recent_frame = None # cv2 frame
    pixel_diffs = []
    hist_diffs = []
    ssims = []


    def __init__(self):
        None


    def calcualte_pixel_difference(self, frame):
        if self.most_recent_frame is None:
            self.most_recent_frame = frame
            return None

        gray_x = cv2.cvtColor(self.most_recent_frame, cv2.COLOR_BGR2GRAY)
        gray_y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_x, gray_y)
        mean_diff = numpy.mean(diff)

        self.pixel_diffs.append(mean_diff)
        return mean_diff


    def calculate_histogram_difference(self, frame):
        if self.most_recent_frame is None:
            self.most_recent_frame = frame
            return None

        gray_x = cv2.cvtColor(self.most_recent_frame, cv2.COLOR_BGR2GRAY)
        gray_y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hist_x = cv2.calcHist([gray_x], [0], None, [256], [0, 256])
        hist_y = cv2.calcHist([gray_y], [0], None, [256], [0, 256])

        # -1 ~ 1: 1 means perfect correlation, 0 means no correlation, -1 means perfect anti correlation
        diff = cv2.compareHist(hist_x, hist_y, cv2.HISTCMP_CORREL)

        self.hist_diffs.append(diff)
        return diff


    def calculate_ssim(self, frame):
        if self.most_recent_frame is None:
            self.most_recent_frame = frame
            return None

        pil_frame = transforms.functional.to_pil_image(frame)
        pil_ref_frame = transforms.functional.to_pil_image(self.most_recent_frame)

        gray_x = transforms.functional.rgb_to_grayscale(pil_frame)
        gray_y = transforms.functional.rgb_to_grayscale(pil_ref_frame)

        pil_to_tensor = transforms.Compose([transforms.PILToTensor()])

        gray_x = pil_to_tensor(gray_x)
        gray_y = pil_to_tensor(gray_y)

        gray_x = torch.unsqueeze(gray_x, dim=0)
        gray_y = torch.unsqueeze(gray_y, dim=0)

        ssim: torch.Tensor = piq.ssim(gray_x, gray_y, data_range=255)
        ssim = ssim.item()

        self.ssims.append(ssim)
        return ssim


if __name__ == "__main__":
    frame_diff_calculator = Frame_Diff_Calculator()
    frame1 = cv2.imread("/home/jin/mnt/github/mmdetection/inference_exp/data/img/000003.jpg")
    frame2 = cv2.imread("/home/jin/mnt/github/mmdetection/inference_exp/data/img/000004.jpg")

    frame_diff_calculator.most_recent_frame = frame1

    pixel_diff = frame_diff_calculator.calcualte_pixel_difference(frame2)
    hist_diff = frame_diff_calculator.calculate_histogram_difference(frame2)
    ssim = frame_diff_calculator.calculate_ssim(frame2)
    print(pixel_diff)
    print(hist_diff)
    print(ssim)

