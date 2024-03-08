import os
from colorama import Fore, Style
import cv2

class My_Visualizer:
    def __init__(self):
        None

    def draw_bboxes(self, img, bboxes1, color1=(0, 255, 0), thickness1=2, bboxes2=None, color2=(0, 0, 255), thickness2=2):
        for bbox in bboxes1:
            left = int(max(bbox[0], 0))
            top = int(max(bbox[1], 0))
            right = int(min(bbox[2], img.shape[1]))
            bottom = int(min(bbox[3], img.shape[0]))
            img = cv2.rectangle(img, (left, top), (right, bottom), color1, thickness1)
        if bboxes2 is not None:
            for bbox in bboxes2:
                left = int(max(bbox[0], 0))
                top = int(max(bbox[1], 0))
                right = int(min(bbox[2], img.shape[1]))
                bottom = int(min(bbox[3], img.shape[0]))
                img = cv2.rectangle(img, (left, top), (right, bottom), color2, thickness2)
        return img


if __name__ == "__main__":
    my_visualizer = My_Visualizer()
    img = cv2.imread("../data/img/000002.jpg")
    img = my_visualizer.draw_bboxes(img, [[100, 100, 200, 200], [300, 300, 400, 400]], color1=(0, 255, 0), thickness1=2, bboxes2=[[200, 200, 300, 300], [300, 300, 2000, 2000]], color2=(0, 0, 255), thickness2=2)
    cv2.imwrite("output.jpg", img)

