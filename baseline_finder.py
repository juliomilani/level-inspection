#%% impor modules
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#%%
def show_images(images,titles=['x','x','x','x','x','x'],color_map='viridis'):
    """
    Show images using matplotlib figures
    """
    fig=plt.figure(figsize=(10,8))
    for i,image in enumerate(images):
        fig.add_subplot(2,3,i+1)
        plt.title(titles[i])
        plt.imshow(image,cmap=color_map)

#%% import all images in images/ to the vector images
images_name = [path for path in os.listdir("images/") if path.endswith('.jpg')]
images_bgr = [cv2.imread(os.path.join("images",image_name)) for image_name in images_name]
images_rgb = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) for im_bgr in images_bgr]
images_gray = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) for im_bgr in images_bgr]
show_images(images_rgb,images_name,'gray')


def get_cap_masks(images):
    lower = np.array([0, 60, 100])
    upper = np.array([59, 130, 170])
    cap_masks = [cv2.inRange(img_rgb, lower, upper) for img_rgb in images]
    show_images(cap_masks,images_name,'gray')
    return cap_masks

def find_cap_bb(cap_masks):
    cap_bbs = []
    images = [img for img in images_rgb]
    for i,cap_mask in enumerate(cap_masks):
        cnts = cv2.findContours(cap_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                print("w:%s, y:%s, w:%s, h:%s i:%s" % (x, y, w, h, i))
                if w > 100:
                    cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
                    cv2.line(images[i], (0, y+h), (images[i].shape[1], y+h), (0, 255, 0), thickness=3)
                    el = cap_mask.copy()[y:y + h, x:x + w]
                    pil_im = Image.fromarray(images_rgb[0])

    show_images(images)

find_cap_bb(get_cap_masks(images_rgb))

plt.show()
