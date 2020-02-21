import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops

#PARAMS
kernel_close = np.ones((5,5),np.uint8)
kernel_gauss = (5,5)

def find_level(img_rgb,img_name=None,out_folder=None,write_files=False):
    image_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(image_gray,kernel_gauss,cv.BORDER_DEFAULT)
    canny = cv.Canny(blurred,10,30)
    if write_files:
        cv.imwrite(out_folder+img_name+"_canny.jpg",canny)

    # sobely = cv.Sobel(blurred,cv.CV_64F,0,1,ksize=5)
    # sobely = sobely>200
    # cv.imwrite(out_folder+img_name+"_sobely.jpg",sobely)

    img_lines = image_gray.copy()
    minLineLength=40
    lines = cv.HoughLinesP(image=canny,rho=1,theta=np.pi/180, threshold=65,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    try:
        a,b,c = lines.shape
    except:
        return 0

    y_min = 600
    for i in range(a):
        p1 = (lines[i][0][0], lines[i][0][1])
        p2 = (lines[i][0][2], lines[i][0][3])
        if(np.abs(p1[1]-p2[1]) < 10 ):
            cv.line(img_lines,p1, p2, (0, 0, 255), 3, cv.LINE_AA)
            ymed = (p1[1]+p2[1])/2
            if ymed < y_min:
                y_min = int(ymed)

    if write_files:
        cv.imwrite(out_folder+img_name+'_hough.jpg',img_lines)

    return y_min

if __name__ == '__main__':
    path_in = "images/with_lighting/"
    path_out = "out/2102/level/"

    images_name = [path for path in os.listdir(path_in) if path.endswith('.jpg')]
    for img_name in images_name:
        img_path = os.path.join(path_in,img_name)
        img_title = os.path.splitext(img_name)[0]
        img = cv.imread(img_path)
        img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        find_level(img_rgb,img_title,path_out,True)
        print(img_name)