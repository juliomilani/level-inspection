import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from level_finder import find_level
from baseline_finder import get_cap_bbox

if __name__ == '__main__':
    path_in = "images/with_lighting"
    path_out = "out/2102/main/"

    images_name = [path for path in os.listdir(path_in) if path.endswith('.jpg')]
    for img_name in images_name:
        img_path = os.path.join(path_in,img_name)
        img_title = os.path.splitext(img_name)[0]

        img = cv.imread(img_path)
        img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        image_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        y_level = find_level(image_gray)
        print(y_level)
        baseline = get_cap_bbox(img_rgb)[2]


        cv.line(img, (0, y_level), (img.shape[1], y_level), (255, 0, 0), thickness=3)
        cv.line(img, (0, baseline), (img.shape[1], baseline), (0, 255, 0), thickness=3)
        cv.putText(img,"{} pxs".format(y_level-baseline),(0,y_level-3),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
        cv.imwrite(path_out+img_title+'out.jpg',img)
        print(path_out+img_title+'out.jpg')
