import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops

#PARAMS
kernel_close = np.ones((5,5),np.uint8)
kernel_gauss = (5,5)


def get_cap_bbox(img_rgb,img_name=None,out_folder=None,write_files=False):
    lower = np.array([0, 60, 100])
    upper = np.array([59, 130, 200])
    img_blurred = cv.GaussianBlur(img_rgb,kernel_gauss,0)

    mask_cap = cv.inRange(img_blurred, lower, upper)
    if write_files:
        cv.imwrite(out_folder+img_name+"_mask.jpg",mask_cap)
    mask_cap = cv.morphologyEx(mask_cap, cv.MORPH_CLOSE, kernel_close)
    mask_cap = cv.morphologyEx(mask_cap, cv.MORPH_OPEN, kernel_close)
    if write_files:
        cv.imwrite(out_folder+img_name+"_maskmorph.jpg",mask_cap)
    props = regionprops(mask_cap)
    if write_files:
        (ymin, xmin, ymax, xmax) = props[0].bbox
        cv.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=3)
        cv.line(img_rgb, (0, ymax), (img_rgb.shape[1], ymax), (0, 255, 0), thickness=3)
        cv.imwrite(out_folder+img_name+"_bbox.jpg",img_rgb)

    return props[0].bbox

if __name__ == '__main__':
    path_in = "images/with_lighting/"
    path_out = "out/2102/"
    images_name = [path for path in os.listdir(path_in) if path.endswith('.jpg')]
    for img_name in images_name:
        img_path = os.path.join(path_in,img_name)
        img_title = os.path.splitext(img_name)[0]
        img = cv.imread(img_path)
        img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        get_cap_bbox(img_rgb,img_title,path_out,write_files=True)
        print(img_name)