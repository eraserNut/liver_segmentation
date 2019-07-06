import cv2 as cv
import os
# from ..config_path import training_root, testing_root
from misc import check_mkdir
import numpy as np

training_root = '/home/chenzhihao/medical_image_proc/datasets/liver_lesion/train'
testing_root = '/home/chenzhihao/medical_image_proc/datasets/liver_lesion/test'
def counter_detection(image):
    # image = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    counter = cv.Canny(gray, 50, 150)
    kernel = np.ones((4,4), np.uint8)
    counter = cv.dilate(counter,kernel,iterations=1)
    return counter
    # dst = cv.bitwise_and(image,image, mask=counter)


if __name__ == '__main__':
    current_root = testing_root
    save_path = os.path.join(current_root, 'counter')
    for img_name in os.listdir(os.path.join(current_root, 'seg')):
        src = cv.imread(os.path.join(current_root, 'seg', img_name))
        counter = counter_detection(src)
        check_mkdir(save_path)
        cv.imwrite(os.path.join(save_path, img_name), counter)